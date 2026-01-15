import OpenAI from "openai";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY!;

// MRL (Matryoshka) configuration - can use 256, 512, 1024, or 3072
// Lower dimensions = faster search, less storage, minimal quality loss
const EMBEDDING_MODEL = "text-embedding-3-small";
const EMBEDDING_DIMENSIONS = 256; // MRL: 3x smaller than default 1536

let _client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!_client) {
    _client = new OpenAI({ apiKey: OPENAI_API_KEY });
  }
  return _client;
}

/**
 * Generate dense embedding using OpenAI text-embedding-3 with MRL
 */
export async function embed(text: string): Promise<number[]> {
  const client = getClient();

  const response = await client.embeddings.create({
    model: EMBEDDING_MODEL,
    input: text,
    dimensions: EMBEDDING_DIMENSIONS,
  });

  if (response.data && response.data.length > 0) {
    return response.data[0].embedding;
  }

  throw new Error("Failed to get embedding from OpenAI");
}

/**
 * Batch embed multiple texts
 */
export async function embedBatch(texts: string[]): Promise<number[][]> {
  const client = getClient();

  const response = await client.embeddings.create({
    model: EMBEDDING_MODEL,
    input: texts,
    dimensions: EMBEDDING_DIMENSIONS,
  });

  if (response.data && response.data.length > 0) {
    return response.data.map((d) => d.embedding);
  }

  throw new Error("Failed to get embeddings from OpenAI");
}

/**
 * Generate BM25 sparse vector from text
 * Uses simple tokenization for Qdrant's sparse vector format
 * Note: Qdrant Cloud can also do this automatically with FastEmbed
 */
export function generateBM25Sparse(text: string): { indices: number[]; values: number[] } {
  // Simple BM25-style tokenization
  const tokens = text
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2);

  // Count term frequencies
  const termFreq = new Map<string, number>();
  for (const token of tokens) {
    termFreq.set(token, (termFreq.get(token) || 0) + 1);
  }

  // Convert to sparse vector format
  // Use hash of token as index (simple approach)
  const indices: number[] = [];
  const values: number[] = [];

  for (const [token, freq] of termFreq) {
    // Simple hash function for token -> index
    let hash = 0;
    for (let i = 0; i < token.length; i++) {
      hash = ((hash << 5) - hash + token.charCodeAt(i)) | 0;
    }
    // Keep index positive and within reasonable range
    const index = Math.abs(hash) % 30000;

    // BM25-style TF score (simplified)
    const tf = freq / (freq + 1.2);

    indices.push(index);
    values.push(tf);
  }

  return { indices, values };
}

// Export configuration for collection creation
export const EMBEDDING_CONFIG = {
  model: EMBEDDING_MODEL,
  dimensions: EMBEDDING_DIMENSIONS,
  provider: "openai",
};
