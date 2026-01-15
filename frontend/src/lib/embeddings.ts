import { CohereClientV2 } from "cohere-ai";

const COHERE_API_KEY = process.env.COHERE_API_KEY!;

// Cohere embed-multilingual-v2.0 - matches existing wikipedia_multimodal collection
const EMBEDDING_MODEL = "embed-multilingual-v2.0";
const EMBEDDING_DIMENSIONS = 768;

let _client: CohereClientV2 | null = null;

function getClient(): CohereClientV2 {
  if (!_client) {
    _client = new CohereClientV2({ token: COHERE_API_KEY });
  }
  return _client;
}

/**
 * Generate dense embedding using Cohere (matches existing 35M collection)
 */
export async function embed(text: string): Promise<number[]> {
  const client = getClient();

  const response = await client.embed({
    model: EMBEDDING_MODEL,
    texts: [text],
    inputType: "search_query",
    embeddingTypes: ["float"],
  });

  const embeddings = response.embeddings;
  if (embeddings && "float" in embeddings && embeddings.float && embeddings.float.length > 0) {
    return embeddings.float[0];
  }

  throw new Error("Failed to get embedding from Cohere");
}

/**
 * Batch embed multiple texts
 */
export async function embedBatch(texts: string[]): Promise<number[][]> {
  const client = getClient();

  const response = await client.embed({
    model: EMBEDDING_MODEL,
    texts,
    inputType: "search_document",
    embeddingTypes: ["float"],
  });

  const embeddings = response.embeddings;
  if (embeddings && "float" in embeddings && embeddings.float) {
    return embeddings.float;
  }

  throw new Error("Failed to get embeddings from Cohere");
}

/**
 * Generate BM25 sparse vector from text
 * For future hybrid search with new collections
 */
export function generateBM25Sparse(text: string): { indices: number[]; values: number[] } {
  const tokens = text
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2);

  const termFreq = new Map<string, number>();
  for (const token of tokens) {
    termFreq.set(token, (termFreq.get(token) || 0) + 1);
  }

  const indices: number[] = [];
  const values: number[] = [];

  termFreq.forEach((freq, token) => {
    let hash = 0;
    for (let i = 0; i < token.length; i++) {
      hash = ((hash << 5) - hash + token.charCodeAt(i)) | 0;
    }
    const index = Math.abs(hash) % 30000;
    const tf = freq / (freq + 1.2);
    indices.push(index);
    values.push(tf);
  });

  return { indices, values };
}

// Export configuration
export const EMBEDDING_CONFIG = {
  model: EMBEDDING_MODEL,
  dimensions: EMBEDDING_DIMENSIONS,
  provider: "cohere",
};
