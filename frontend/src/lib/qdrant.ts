import { embed } from "./embeddings";

const QDRANT_URL = process.env.QDRANT_URL!;
const QDRANT_API_KEY = process.env.QDRANT_API_KEY!;

// Collections
const COLLECTIONS = {
  world: "wikipedia_multimodal",
  articles: "user_articles",
  feedback: "user_feedback",
  linkup: "linkup_cache",
};

async function qdrantRequest(endpoint: string, method: string = "GET", body?: unknown) {
  const response = await fetch(`${QDRANT_URL}${endpoint}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      "api-key": QDRANT_API_KEY,
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    throw new Error(`Qdrant API error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

export interface WikiSection {
  page_id: string;
  page_title: string;
  section_text: string;
  url: string;
  score: number;
}

export interface Article {
  article_id: string;
  user_id: string;
  title: string;
  content: string;
  topic_a: string;
  topic_b: string;
  source_page_ids: string[];
  source_urls: string[];
  created_at: number;
}

export interface Feedback {
  feedback_id: string;
  user_id: string;
  article_id: string;
  feedback_type: string;
  feedback_text: string;
  rating: number;
  created_at: number;
}

// Search Wikipedia for grounding context
export async function searchWorld(query: string, topK: number = 8): Promise<WikiSection[]> {
  const queryVector = await embed(query);

  const result = await qdrantRequest(`/collections/${COLLECTIONS.world}/points/query`, "POST", {
    query: queryVector,
    using: "text",
    limit: topK,
    params: { indexed_only: true },
    with_payload: true,
  });

  return (result.result?.points || []).map((point: { payload?: Record<string, unknown>; score?: number }) => ({
    page_id: String(point.payload?.title || "").replace(/ /g, "_"),
    page_title: String(point.payload?.title || "Unknown"),
    section_text: String(point.payload?.text || ""),
    url: String(point.payload?.url || `https://en.wikipedia.org/wiki/${point.payload?.title || ""}`),
    score: point.score || 0,
  }));
}

// Find cached article by topic combination
export async function findCachedArticle(
  topicA: string,
  topicB: string,
  userId: string = "default"
): Promise<Article | null> {
  const topicANorm = topicA.replace(/ /g, "_");
  const topicBNorm = topicB.replace(/ /g, "_");

  try {
    // Try forward order
    let result = await qdrantRequest(`/collections/${COLLECTIONS.articles}/points/scroll`, "POST", {
      filter: {
        must: [
          { key: "topic_a", match: { value: topicANorm } },
          { key: "topic_b", match: { value: topicBNorm } },
        ],
      },
      limit: 1,
      with_payload: true,
    });

    if (result.result?.points?.length > 0) {
      return result.result.points[0].payload as Article;
    }

    // Try reverse order
    result = await qdrantRequest(`/collections/${COLLECTIONS.articles}/points/scroll`, "POST", {
      filter: {
        must: [
          { key: "topic_a", match: { value: topicBNorm } },
          { key: "topic_b", match: { value: topicANorm } },
        ],
      },
      limit: 1,
      with_payload: true,
    });

    if (result.result?.points?.length > 0) {
      return result.result.points[0].payload as Article;
    }
  } catch (e) {
    console.error("Cache lookup error:", e);
  }

  return null;
}

// Store a generated article
export async function storeArticle(
  userId: string,
  title: string,
  content: string,
  topicA: string,
  topicB: string,
  sourcePageIds: string[] = [],
  sourceUrls: string[] = []
): Promise<Article> {
  const articleId = crypto.randomUUID().replace(/-/g, "");

  const article: Article = {
    article_id: articleId,
    user_id: userId,
    title,
    content,
    topic_a: topicA.replace(/ /g, "_"),
    topic_b: topicB.replace(/ /g, "_"),
    source_page_ids: sourcePageIds,
    source_urls: sourceUrls,
    created_at: Math.floor(Date.now() / 1000),
  };

  const vector = await embed(`${title} ${content.slice(0, 500)}`);

  await qdrantRequest(`/collections/${COLLECTIONS.articles}/points`, "PUT", {
    points: [
      {
        id: articleId,
        vector,
        payload: article,
      },
    ],
  });

  return article;
}

// Get user preferences from feedback
export async function getUserPreferences(
  userId: string,
  feedbackType?: string,
  topK: number = 10
): Promise<Feedback[]> {
  const mustConditions: Array<{ key: string; match: { value: string } }> = [
    { key: "user_id", match: { value: userId } },
  ];

  if (feedbackType) {
    mustConditions.push({ key: "feedback_type", match: { value: feedbackType } });
  }

  try {
    const result = await qdrantRequest(`/collections/${COLLECTIONS.feedback}/points/scroll`, "POST", {
      filter: { must: mustConditions },
      limit: topK,
      with_payload: true,
    });

    return (result.result?.points || []).map((p: { payload: Feedback }) => p.payload);
  } catch {
    return [];
  }
}

// Store user feedback
export async function storeFeedback(
  userId: string,
  articleId: string,
  feedbackText: string,
  feedbackType: string = "text_style",
  rating: number = 3
): Promise<Feedback> {
  const feedbackId = crypto.randomUUID().replace(/-/g, "");

  const feedback: Feedback = {
    feedback_id: feedbackId,
    user_id: userId,
    article_id: articleId,
    feedback_type: feedbackType,
    feedback_text: feedbackText,
    rating,
    created_at: Math.floor(Date.now() / 1000),
  };

  const vector = await embed(feedbackText);

  await qdrantRequest(`/collections/${COLLECTIONS.feedback}/points`, "PUT", {
    points: [
      {
        id: feedbackId,
        vector,
        payload: feedback,
      },
    ],
  });

  return feedback;
}

// Cache Linkup results
export async function cacheLinkupResult(
  query: string,
  url: string,
  title: string,
  snippet: string,
  content: string,
  ttlHours: number = 24
): Promise<void> {
  const resultId = crypto.randomUUID().replace(/-/g, "");
  const now = Math.floor(Date.now() / 1000);

  const result = {
    result_id: resultId,
    query,
    url,
    title,
    snippet,
    content,
    fetched_at: now,
    expires_at: now + ttlHours * 3600,
  };

  const vector = await embed(`${title} ${snippet} ${content.slice(0, 500)}`);

  await qdrantRequest(`/collections/${COLLECTIONS.linkup}/points`, "PUT", {
    points: [
      {
        id: resultId,
        vector,
        payload: result,
      },
    ],
  });
}

// Search Linkup cache
export async function searchLinkupCache(query: string, topK: number = 5): Promise<Array<{
  url: string;
  title: string;
  snippet: string;
  content: string;
  score: number;
}>> {
  const queryVector = await embed(query);
  const now = Math.floor(Date.now() / 1000);

  try {
    const result = await qdrantRequest(`/collections/${COLLECTIONS.linkup}/points/query`, "POST", {
      query: queryVector,
      limit: topK,
      with_payload: true,
    });

    return (result.result?.points || [])
      .filter((p: { payload?: { expires_at?: number } }) => (p.payload?.expires_at || 0) > now)
      .map((p: { payload?: Record<string, unknown>; score?: number }) => ({
        url: String(p.payload?.url || ""),
        title: String(p.payload?.title || ""),
        snippet: String(p.payload?.snippet || ""),
        content: String(p.payload?.content || ""),
        score: p.score || 0,
      }));
  } catch {
    return [];
  }
}
