import { embed, generateBM25Sparse, EMBEDDING_CONFIG } from "./embeddings";

const QDRANT_URL = process.env.QDRANT_URL!;
const QDRANT_API_KEY = process.env.QDRANT_API_KEY!;

// Collections
const COLLECTIONS = {
  world: "wikipedia_multimodal",
  articles: "user_articles",
  feedback: "user_feedback",
  linkup: "linkup_cache",
};

// =============================================================================
// Metrics Tracking (for observability)
// =============================================================================

interface QueryMetric {
  collection: string;
  latencyMs: number;
  timestamp: number;
}

const metricsStore = {
  queries: [] as QueryMetric[],
  cacheHits: 0,
  cacheMisses: 0,
};

export function recordQueryMetric(collection: string, latencyMs: number) {
  metricsStore.queries.push({
    collection,
    latencyMs,
    timestamp: Date.now(),
  });
  // Keep only last 1000 queries
  if (metricsStore.queries.length > 1000) {
    metricsStore.queries = metricsStore.queries.slice(-1000);
  }
}

export function recordCacheHit(hit: boolean) {
  if (hit) metricsStore.cacheHits++;
  else metricsStore.cacheMisses++;
}

export function getMetrics() {
  const recentQueries = metricsStore.queries.filter(
    (q) => q.timestamp > Date.now() - 60000 // Last minute
  );
  const latencies = recentQueries.map((q) => q.latencyMs).sort((a, b) => a - b);

  return {
    queryLatency: {
      p50: latencies[Math.floor(latencies.length * 0.5)] || 0,
      p95: latencies[Math.floor(latencies.length * 0.95)] || 0,
      p99: latencies[Math.floor(latencies.length * 0.99)] || 0,
      count: latencies.length,
    },
    cacheHitRate:
      metricsStore.cacheHits + metricsStore.cacheMisses > 0
        ? metricsStore.cacheHits / (metricsStore.cacheHits + metricsStore.cacheMisses)
        : 0,
    cacheStats: {
      hits: metricsStore.cacheHits,
      misses: metricsStore.cacheMisses,
    },
    embeddingConfig: EMBEDDING_CONFIG,
  };
}

// =============================================================================
// Core Qdrant API
// =============================================================================

async function qdrantRequest(endpoint: string, method: string = "GET", body?: unknown) {
  const start = Date.now();
  const response = await fetch(`${QDRANT_URL}${endpoint}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      "api-key": QDRANT_API_KEY,
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  const latency = Date.now() - start;

  // Extract collection name from endpoint for metrics
  const collectionMatch = endpoint.match(/\/collections\/([^/]+)/);
  if (collectionMatch) {
    recordQueryMetric(collectionMatch[1], latency);
  }

  if (!response.ok) {
    throw new Error(`Qdrant API error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

// =============================================================================
// Types
// =============================================================================

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

export interface SearchFilters {
  minScore?: number;
  categories?: string[];
  excludePageIds?: string[];
}

// =============================================================================
// Re-ranking (Score Boosting)
// Customer Evidence: Re-ranking brings explainability and control to semantic search
// =============================================================================

export function rerankResults(
  results: WikiSection[],
  topicA: string,
  topicB: string
): WikiSection[] {
  const a = topicA.toLowerCase();
  const b = topicB.toLowerCase();

  return results
    .map((r) => {
      let boost = 1.0;
      const titleLower = r.page_title.toLowerCase();
      const textLower = r.section_text.toLowerCase();

      // Boost if topic A appears in title
      if (titleLower.includes(a)) boost += 0.15;

      // Boost if topic B appears in title
      if (titleLower.includes(b)) boost += 0.15;

      // Boost if BOTH topics appear in content
      if (textLower.includes(a) && textLower.includes(b)) boost += 0.25;

      // Boost if topic appears at start of title
      if (titleLower.startsWith(a) || titleLower.startsWith(b)) boost += 0.1;

      return {
        ...r,
        score: r.score * boost,
      };
    })
    .sort((x, y) => y.score - x.score);
}

// =============================================================================
// Search with Payload Filtering (HNSW In-Graph Filtering)
// Customer Evidence: Qdrant's HNSW does metadata filtering within the search step
// =============================================================================

export async function searchWorld(
  query: string,
  topK: number = 8,
  filters?: SearchFilters
): Promise<WikiSection[]> {
  const queryVector = await embed(query);

  // Build filter conditions
  const mustConditions: unknown[] = [];
  const mustNotConditions: unknown[] = [];

  if (filters?.categories?.length) {
    mustConditions.push({
      key: "category",
      match: { any: filters.categories },
    });
  }

  if (filters?.excludePageIds?.length) {
    mustNotConditions.push({
      key: "title",
      match: { any: filters.excludePageIds },
    });
  }

  const filter =
    mustConditions.length || mustNotConditions.length
      ? {
          ...(mustConditions.length && { must: mustConditions }),
          ...(mustNotConditions.length && { must_not: mustNotConditions }),
        }
      : undefined;

  const result = await qdrantRequest(`/collections/${COLLECTIONS.world}/points/query`, "POST", {
    query: queryVector,
    using: "text",
    limit: topK,
    params: { indexed_only: true },
    with_payload: true,
    ...(filter && { filter }),
  });

  let sections = (result.result?.points || []).map(
    (point: { payload?: Record<string, unknown>; score?: number }) => ({
      page_id: String(point.payload?.title || "").replace(/ /g, "_"),
      page_title: String(point.payload?.title || "Unknown"),
      section_text: String(point.payload?.text || ""),
      url: String(point.payload?.url || `https://en.wikipedia.org/wiki/${point.payload?.title || ""}`),
      score: point.score || 0,
    })
  );

  // Apply minimum score filter client-side
  if (filters?.minScore) {
    sections = sections.filter((s: WikiSection) => s.score >= (filters.minScore || 0));
  }

  return sections;
}

// =============================================================================
// Recommendation API (Cross-Topic Connections)
// Customer Evidence: Zepto advised to use Recommendation API to avoid network latency
// =============================================================================

export async function findConnectionsViaRecommend(
  topicAResults: WikiSection[],
  topicBResults: WikiSection[],
  limit: number = 5
): Promise<WikiSection[]> {
  // Get point IDs from results (using page titles as lookup)
  const positiveExamples: string[] = [];

  // Add top results from both topics as positive examples
  const topA = topicAResults.slice(0, 2);
  const topB = topicBResults.slice(0, 2);

  for (const section of [...topA, ...topB]) {
    positiveExamples.push(section.page_id);
  }

  if (positiveExamples.length < 2) {
    return []; // Not enough examples for recommendation
  }

  try {
    // First, we need to find the actual point IDs
    // Search for points by title to get their IDs
    const pointIds: string[] = [];

    for (const pageId of positiveExamples) {
      const scrollResult = await qdrantRequest(
        `/collections/${COLLECTIONS.world}/points/scroll`,
        "POST",
        {
          filter: {
            must: [{ key: "title", match: { value: pageId.replace(/_/g, " ") } }],
          },
          limit: 1,
          with_payload: false,
        }
      );

      if (scrollResult.result?.points?.[0]?.id) {
        pointIds.push(scrollResult.result.points[0].id);
      }
    }

    if (pointIds.length < 2) {
      return [];
    }

    // Use Qdrant's recommend API
    const result = await qdrantRequest(
      `/collections/${COLLECTIONS.world}/points/recommend`,
      "POST",
      {
        positive: pointIds,
        limit,
        using: "text",
        with_payload: true,
      }
    );

    return (result.result || []).map(
      (point: { payload?: Record<string, unknown>; score?: number }) => ({
        page_id: String(point.payload?.title || "").replace(/ /g, "_"),
        page_title: String(point.payload?.title || "Unknown"),
        section_text: String(point.payload?.text || ""),
        url: String(
          point.payload?.url || `https://en.wikipedia.org/wiki/${point.payload?.title || ""}`
        ),
        score: point.score || 0,
      })
    );
  } catch (error) {
    console.error("Recommendation API error:", error);
    return [];
  }
}

// =============================================================================
// TTL Cleanup for Linkup Cache
// =============================================================================

export async function cleanupExpiredCache(): Promise<number> {
  const now = Math.floor(Date.now() / 1000);

  try {
    const result = await qdrantRequest(`/collections/${COLLECTIONS.linkup}/points/delete`, "POST", {
      filter: {
        must: [
          {
            key: "expires_at",
            range: { lt: now },
          },
        ],
      },
    });

    return result.result?.deleted || 0;
  } catch (error) {
    console.error("TTL cleanup error:", error);
    return 0;
  }
}

// =============================================================================
// Hybrid Search (Dense + Sparse)
// Customer Evidence: Mixpeak migrated from MongoDB citing easier hybrid search
// =============================================================================

export async function searchWorldHybrid(
  query: string,
  topK: number = 8,
  options?: {
    filters?: SearchFilters;
    useSparse?: boolean;
    topicA?: string;
    topicB?: string;
  }
): Promise<WikiSection[]> {
  const queryVector = await embed(query);

  // Build request based on whether we have sparse vectors
  let searchRequest: Record<string, unknown>;

  if (options?.useSparse) {
    // Hybrid search with RRF fusion
    const sparseVector = generateBM25Sparse(query);

    searchRequest = {
      prefetch: [
        {
          query: queryVector,
          using: "dense",
          limit: topK * 2,
        },
        {
          query: {
            indices: sparseVector.indices,
            values: sparseVector.values,
          },
          using: "sparse",
          limit: topK * 2,
        },
      ],
      query: { fusion: "rrf" },
      limit: topK,
      with_payload: true,
    };
  } else {
    // Dense-only search (current behavior)
    searchRequest = {
      query: queryVector,
      using: "text",
      limit: topK,
      params: { indexed_only: true },
      with_payload: true,
    };
  }

  const result = await qdrantRequest(
    `/collections/${COLLECTIONS.world}/points/query`,
    "POST",
    searchRequest
  );

  let sections = (result.result?.points || []).map(
    (point: { payload?: Record<string, unknown>; score?: number }) => ({
      page_id: String(point.payload?.title || "").replace(/ /g, "_"),
      page_title: String(point.payload?.title || "Unknown"),
      section_text: String(point.payload?.text || ""),
      url: String(point.payload?.url || `https://en.wikipedia.org/wiki/${point.payload?.title || ""}`),
      score: point.score || 0,
    })
  );

  // Apply re-ranking if topics provided
  if (options?.topicA && options?.topicB) {
    sections = rerankResults(sections, options.topicA, options.topicB);
  }

  // Apply minimum score filter
  if (options?.filters?.minScore) {
    sections = sections.filter((s: WikiSection) => s.score >= (options.filters?.minScore || 0));
  }

  return sections;
}

// =============================================================================
// Collection Info (for metrics dashboard)
// =============================================================================

export async function getCollectionInfo(
  collectionName: string
): Promise<{ points: number; status: string } | null> {
  try {
    const result = await qdrantRequest(`/collections/${collectionName}`, "GET");
    return {
      points: result.result?.points_count || 0,
      status: result.result?.status || "unknown",
    };
  } catch {
    return null;
  }
}

export async function getAllCollectionsInfo(): Promise<Record<string, { points: number; status: string }>> {
  const info: Record<string, { points: number; status: string }> = {};

  for (const [key, name] of Object.entries(COLLECTIONS)) {
    const collectionInfo = await getCollectionInfo(name);
    if (collectionInfo) {
      info[key] = collectionInfo;
    }
  }

  return info;
}

// =============================================================================
// Existing Functions (unchanged)
// =============================================================================

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
      recordCacheHit(true);
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
      recordCacheHit(true);
      return result.result.points[0].payload as Article;
    }

    recordCacheHit(false);
  } catch (e) {
    console.error("Cache lookup error:", e);
    recordCacheHit(false);
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
export async function searchLinkupCache(
  query: string,
  topK: number = 5
): Promise<
  Array<{
    url: string;
    title: string;
    snippet: string;
    content: string;
    score: number;
  }>
> {
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
