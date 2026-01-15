import { CohereClient } from "cohere-ai";

const COHERE_API_KEY = process.env.COHERE_API_KEY!;
const EMBEDDING_MODEL = "embed-multilingual-v2.0";

let _client: CohereClient | null = null;

function getClient(): CohereClient {
  if (!_client) {
    _client = new CohereClient({ token: COHERE_API_KEY });
  }
  return _client;
}

export async function embed(text: string): Promise<number[]> {
  const client = getClient();

  const response = await client.embed({
    texts: [text],
    model: EMBEDDING_MODEL,
    inputType: "search_query",
  });

  if (response.embeddings && Array.isArray(response.embeddings) && response.embeddings.length > 0) {
    const embedding = response.embeddings[0];
    if (Array.isArray(embedding)) {
      return embedding;
    }
  }

  throw new Error("Failed to get embedding from Cohere");
}

export async function embedBatch(texts: string[]): Promise<number[][]> {
  const client = getClient();

  const response = await client.embed({
    texts,
    model: EMBEDDING_MODEL,
    inputType: "search_query",
  });

  if (response.embeddings && Array.isArray(response.embeddings)) {
    return response.embeddings as number[][];
  }

  throw new Error("Failed to get embeddings from Cohere");
}
