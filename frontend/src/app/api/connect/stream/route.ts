import { NextRequest } from "next/server";
import {
  searchWorld,
  findCachedArticle,
  storeArticle,
  getUserPreferences,
} from "@/lib/qdrant";
import { generateContentStream, buildArticlePrompt } from "@/lib/gemini";
import { searchAndCache, isAvailable as linkupAvailable, formatForGrounding } from "@/lib/linkup";

function sendEvent(eventType: string, data: Record<string, unknown>): string {
  return `data: ${JSON.stringify({ type: eventType, ...data })}\n\n`;
}

function sendDecision(decision: string, reasoning: string, action: string): string {
  return sendEvent("decision", { decision, reasoning, action });
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const topicA = body.topicA as string;
  const topicB = body.topicB as string;
  const userId = (body.userId as string) || "default";

  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      const send = (data: string) => controller.enqueue(encoder.encode(data));

      try {
        // Decision: Query Routing
        send(sendDecision(
          "Query Routing",
          `Input topics: '${topicA}' and '${topicB}'. Need to determine optimal retrieval strategy.`,
          "Route to multi-collection search: (1) Check cache, (2) Wikipedia semantic search, (3) Cross-topic bridging, (4) Optional web fallback"
        ));

        // Step 1: Check cache
        send(sendEvent("step", {
          step: "cache",
          status: "running",
          message: "Checking Qdrant user_articles collection...",
          detail: `Looking for cached article matching topics: ${topicA} ↔ ${topicB}`,
        }));

        const cached = await findCachedArticle(topicA, topicB, userId);

        if (cached) {
          send(sendEvent("step", {
            step: "cache",
            status: "done",
            message: "Cache HIT! Found existing article",
            detail: `Article ID: ${cached.article_id.slice(0, 8)}...`,
            results: [`Title: ${cached.title}`, `Sources: ${cached.source_urls.length} Wikipedia pages`],
          }));

          // Get preferences
          send(sendEvent("step", {
            step: "preferences",
            status: "running",
            message: "Loading user preferences from Qdrant...",
            detail: `Querying user_feedback collection for user: ${userId}`,
          }));

          const preferences = await getUserPreferences(userId, undefined, 10);
          const appliedPreferences = preferences.slice(0, 5).map((p) => ({
            type: p.feedback_type,
            text: p.feedback_text,
            rating: p.rating,
          }));

          send(sendEvent("step", {
            step: "preferences",
            status: "done",
            message: preferences.length > 0 ? `Found ${preferences.length} stored preferences` : "No preferences found",
            detail: preferences.length > 0 ? "Preferences would be applied on regeneration" : "Give feedback to personalize future articles",
            results: preferences.slice(0, 3).map((p) => `${p.feedback_type}: ${p.feedback_text}`),
          }));

          send(sendEvent("article", {
            articleId: cached.article_id,
            title: cached.title,
            content: cached.content,
            sources: cached.source_urls,
            cached: true,
          }));

          send(sendEvent("complete", { cached: true, appliedPreferences }));
          controller.close();
          return;
        }

        send(sendEvent("step", {
          step: "cache",
          status: "done",
          message: "Cache MISS - Generating new article",
          detail: "No existing article found for this topic combination",
        }));

        // Decision: Memory Strategy
        send(sendDecision(
          "Memory Strategy",
          "No cached article exists. Must retrieve grounding context from multiple memory layers.",
          "Execute parallel retrieval: WORLD (Wikipedia 35M) for facts, USER_FEEDBACK for style preferences, optionally WORLD_DYNAMIC (Linkup) for fresh content"
        ));

        // Step 2: Search Topic A
        send(sendEvent("step", {
          step: "search_a",
          status: "running",
          message: "Searching Qdrant wikipedia_multimodal collection...",
          detail: `Query: "${topicA}" | Collection: 35M+ Wikipedia article sections | Using: Cohere embeddings`,
        }));

        const resultsA = await searchWorld(topicA, 4);
        if (!resultsA.length) {
          send(sendEvent("error", { message: `No Wikipedia info for '${topicA}'` }));
          controller.close();
          return;
        }

        const scoresA = resultsA.slice(0, 3).map((r) => r.score.toFixed(3)).join(", ");
        send(sendEvent("step", {
          step: "search_a",
          status: "done",
          message: `Found ${resultsA.length} relevant sections for "${topicA}"`,
          detail: `Top similarity scores: ${scoresA}`,
          results: resultsA.slice(0, 3).map((r) => `${r.page_title}: ${r.section_text.slice(0, 60)}...`),
        }));

        // Step 3: Search Topic B
        send(sendEvent("step", {
          step: "search_b",
          status: "running",
          message: "Searching Qdrant wikipedia_multimodal collection...",
          detail: `Query: "${topicB}" | Using same 35M article index`,
        }));

        const resultsB = await searchWorld(topicB, 4);
        if (!resultsB.length) {
          send(sendEvent("error", { message: `No Wikipedia info for '${topicB}'` }));
          controller.close();
          return;
        }

        const scoresB = resultsB.slice(0, 3).map((r) => r.score.toFixed(3)).join(", ");
        send(sendEvent("step", {
          step: "search_b",
          status: "done",
          message: `Found ${resultsB.length} relevant sections for "${topicB}"`,
          detail: `Top similarity scores: ${scoresB}`,
          results: resultsB.slice(0, 3).map((r) => `${r.page_title}: ${r.section_text.slice(0, 60)}...`),
        }));

        // Step 4: Search connections
        const connectionQuery = `${topicA} ${topicB} relationship`;
        send(sendEvent("step", {
          step: "connections",
          status: "running",
          message: "Finding cross-topic connections in Qdrant...",
          detail: `Query: "${connectionQuery}" | Looking for bridging concepts`,
        }));

        const connectionResults = await searchWorld(connectionQuery, 3);
        send(sendEvent("step", {
          step: "connections",
          status: "done",
          message: `Found ${connectionResults.length} potential connection points`,
          detail: "These sections may contain shared themes or relationships",
          results: connectionResults.slice(0, 2).map((r) => r.url),
        }));

        // Decision: Linkup routing
        const avgScore = (
          resultsA.slice(0, 2).reduce((sum, r) => sum + r.score, 0) +
          resultsB.slice(0, 2).reduce((sum, r) => sum + r.score, 0)
        ) / 4;

        const needsLinkup = avgScore < 0.85 || linkupAvailable();
        send(sendDecision(
          "Linkup Routing",
          `Wikipedia relevance: avg score ${avgScore.toFixed(3)}. ${needsLinkup ? "Scores below 0.85 threshold or Linkup available - checking web for supplementary content." : "Wikipedia coverage sufficient, skipping web search."}`,
          needsLinkup && linkupAvailable() ? "Query Linkup API for recent web content" : "Skip web search, proceed with Wikipedia grounding only"
        ));

        // Step 5: Linkup search
        let linkupContext = "";
        if (linkupAvailable()) {
          send(sendEvent("step", {
            step: "linkup",
            status: "running",
            message: "Searching Linkup for recent web content...",
            detail: `Query: "${topicA} ${topicB} connection" | For topics not well covered in Wikipedia`,
          }));

          try {
            const linkupResults = await searchAndCache(`${topicA} ${topicB} connection`, false);
            if (linkupResults.length) {
              linkupContext = formatForGrounding(linkupResults.slice(0, 2));
              send(sendEvent("step", {
                step: "linkup",
                status: "done",
                message: `Found ${linkupResults.length} fresh web results`,
                detail: "Results cached in Qdrant linkup_cache collection (24h TTL)",
                results: linkupResults.slice(0, 2).map((r) => r.title.slice(0, 50)),
              }));
            } else {
              send(sendEvent("step", {
                step: "linkup",
                status: "done",
                message: "No additional web results needed",
                detail: "Wikipedia coverage sufficient",
              }));
            }
          } catch (e) {
            send(sendEvent("step", {
              step: "linkup",
              status: "done",
              message: "Linkup search skipped",
              detail: String(e).slice(0, 50),
            }));
          }
        } else {
          send(sendEvent("step", {
            step: "linkup",
            status: "done",
            message: "Linkup not configured",
            detail: "Set LINKUP_API_KEY to enable fresh web search fallback",
          }));
        }

        // Step 6: Load preferences
        send(sendEvent("step", {
          step: "preferences",
          status: "running",
          message: "Querying Qdrant user_feedback collection...",
          detail: `User: ${userId} | Looking for writing style preferences`,
        }));

        const preferences = await getUserPreferences(userId, undefined, 10);
        let prefInstructions = "";

        if (preferences.length) {
          const textPrefs = preferences.filter((p) => p.feedback_type === "text_style");
          const formatPrefs = preferences.filter((p) => p.feedback_type === "format");
          const contentPrefs = preferences.filter((p) => p.feedback_type === "content");

          const prefParts: string[] = [];
          if (textPrefs.length) {
            prefParts.push(`Writing style: ${textPrefs.slice(0, 3).map((p) => p.feedback_text).join("; ")}`);
          }
          if (formatPrefs.length) {
            prefParts.push(`Format: ${formatPrefs.slice(0, 2).map((p) => p.feedback_text).join("; ")}`);
          }
          if (contentPrefs.length) {
            prefParts.push(`Content: ${contentPrefs.slice(0, 2).map((p) => p.feedback_text).join("; ")}`);
          }

          if (prefParts.length) {
            prefInstructions = "\n\n**USER PREFERENCES (apply these to your writing):**\n" + prefParts.join("\n");
          }

          send(sendEvent("step", {
            step: "preferences",
            status: "done",
            message: `Applying ${preferences.length} user preferences`,
            detail: "These will customize the article's writing style",
            results: preferences.slice(0, 3).map((p) =>
              `${"★".repeat(p.rating)}${"☆".repeat(5 - p.rating)} ${p.feedback_type}: ${p.feedback_text.slice(0, 40)}...`
            ),
          }));

          send(sendDecision(
            "Preference Application",
            `Found ${preferences.length} stored preferences. Text style: ${textPrefs.length}, Format: ${formatPrefs.length}, Content: ${contentPrefs.length}.`,
            `Injecting preference instructions into LLM prompt: ${prefParts[0]?.slice(0, 50) || "none"}...`
          ));
        } else {
          send(sendEvent("step", {
            step: "preferences",
            status: "done",
            message: "No stored preferences found",
            detail: "Give feedback on this article to personalize future generations!",
          }));

          send(sendDecision(
            "Preference Application",
            "No user preferences in USER_FEEDBACK collection. Using default writing style.",
            "Generate with standard Wikipedia-style formatting. User can provide feedback to personalize future outputs."
          ));
        }

        // Step 7: Build context
        send(sendEvent("step", {
          step: "context",
          status: "running",
          message: "Building grounding context from Qdrant + Linkup...",
          detail: "Combining Wikipedia sections into structured prompt",
        }));

        const contextA = resultsA.slice(0, 2).map((s) => s.section_text.slice(0, 300)).join("\n");
        const contextB = resultsB.slice(0, 2).map((s) => s.section_text.slice(0, 300)).join("\n");
        const contextConn = connectionResults.slice(0, 2).map((s) => s.section_text.slice(0, 300)).join("\n");

        const totalContextChars = contextA.length + contextB.length + contextConn.length;
        send(sendEvent("step", {
          step: "context",
          status: "done",
          message: `Context ready: ${totalContextChars.toLocaleString()} characters`,
          detail: `From ${resultsA.slice(0, 2).length + resultsB.slice(0, 2).length + connectionResults.slice(0, 2).length} Wikipedia sections`,
          results: [
            `Topic A context: ${contextA.length.toLocaleString()} chars`,
            `Topic B context: ${contextB.length.toLocaleString()} chars`,
            `Connection context: ${contextConn.length.toLocaleString()} chars`,
            `User preferences: ${prefInstructions ? "YES - will apply" : "None"}`,
          ],
        }));

        // Step 8: Generate article
        send(sendEvent("step", {
          step: "generate",
          status: "running",
          message: "Generating article with Gemini...",
          detail: "Grounded generation using Qdrant vectors + Linkup web search",
        }));

        const prompt = buildArticlePrompt(topicA, topicB, contextA, contextB, contextConn, prefInstructions);

        let articleContent = "";
        try {
          for await (const chunk of generateContentStream(prompt)) {
            articleContent += chunk;
            send(sendEvent("content", { chunk }));
          }
        } catch (e) {
          send(sendEvent("error", { message: `Generation error: ${String(e)}` }));
          controller.close();
          return;
        }

        // Extract title
        const lines = articleContent.trim().split("\n");
        let title = `The Connection Between ${topicA} and ${topicB}`;
        for (const line of lines) {
          if (line.startsWith("# ")) {
            title = line.slice(2).trim();
            break;
          }
        }

        const wordCount = articleContent.split(/\s+/).length;
        send(sendEvent("step", {
          step: "generate",
          status: "done",
          message: `Generated ${wordCount.toLocaleString()} word article`,
          detail: title.length > 50 ? `Title extracted: "${title.slice(0, 50)}..."` : `Title: "${title}"`,
        }));

        // Decision: Storage
        send(sendDecision(
          "Memory Storage",
          `Article generated successfully (${wordCount} words). Need to persist for future retrieval and caching.`,
          `Store in USER_ARTICLES collection with: (1) Cohere embedding of title+content for semantic search, (2) topic_a/topic_b fields for exact-match cache lookup, (3) source_page_ids linking back to Wikipedia`
        ));

        // Step 9: Store article
        send(sendEvent("step", {
          step: "store",
          status: "running",
          message: "Storing article in Qdrant user_articles...",
          detail: `Embedding article with Cohere | User: ${userId}`,
        }));

        const sourceUrls = Array.from(new Set([
          ...resultsA.map((s) => s.url),
          ...resultsB.map((s) => s.url),
          ...connectionResults.map((s) => s.url),
        ])).slice(0, 6);

        const sourcePageIds = Array.from(new Set([
          ...resultsA.map((s) => s.page_id),
          ...resultsB.map((s) => s.page_id),
          ...connectionResults.map((s) => s.page_id),
        ]));

        const article = await storeArticle(
          userId,
          title,
          articleContent,
          topicA,
          topicB,
          sourcePageIds,
          sourceUrls
        );

        send(sendEvent("step", {
          step: "store",
          status: "done",
          message: `Article saved! ID: ${article.article_id.slice(0, 8)}...`,
          detail: `Stored with ${sourceUrls.length} source URLs | ${sourcePageIds.length} page references`,
          results: [
            "Collection: user_articles",
            `Topics: ${topicA} ↔ ${topicB}`,
            "Will be cached for future requests",
          ],
        }));

        // Final response
        const appliedPreferences = preferences.slice(0, 5).map((p) => ({
          type: p.feedback_type,
          text: p.feedback_text,
          rating: p.rating,
        }));

        send(sendEvent("article", {
          articleId: article.article_id,
          title,
          content: articleContent,
          sources: sourceUrls,
          cached: false,
        }));

        send(sendEvent("complete", {
          cached: false,
          appliedPreferences,
          pageIds: sourcePageIds,
        }));

      } catch (e) {
        send(sendEvent("error", { message: `Server error: ${String(e)}` }));
      }

      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
