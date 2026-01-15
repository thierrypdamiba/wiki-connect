import { NextResponse } from "next/server";

const QDRANT_URL = process.env.QDRANT_URL!;
const QDRANT_API_KEY = process.env.QDRANT_API_KEY!;

const EXPERIMENTS_COLLECTION = "ingestion_experiments";

interface Experiment {
  experiment_id: string;
  config: {
    name: string;
    embedding_provider: string;
    embedding_model: string;
    dimensions: number;
    use_mrl: boolean;
    use_quantization: boolean;
    use_sparse: boolean;
    batch_size: number;
  };
  started_at: string;
  completed_at: string | null;
  status: string;
  total_texts: number;
  total_vectors: number;
  total_batches: number;
  total_embedding_time_ms: number;
  total_upsert_time_ms: number;
  total_time_ms: number;
  avg_embedding_time_per_batch_ms: number;
  avg_upsert_time_per_batch_ms: number;
  avg_embedding_time_per_vector_ms: number;
  vectors_per_second: number;
  estimated_vector_size_bytes: number;
  estimated_total_size_mb: number;
  search_latency_p50_ms: number | null;
  search_latency_p95_ms: number | null;
  recall_at_10: number | null;
  errors: string[];
}

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
    const text = await response.text();
    throw new Error(`Qdrant API error: ${response.status} - ${text}`);
  }

  return response.json();
}

export async function GET() {
  try {
    // Check if collection exists
    let collectionExists = false;
    try {
      await qdrantRequest(`/collections/${EXPERIMENTS_COLLECTION}`, "GET");
      collectionExists = true;
    } catch {
      collectionExists = false;
    }

    if (!collectionExists) {
      return NextResponse.json({
        experiments: [],
        summary: {
          total: 0,
          completed: 0,
          failed: 0,
          running: 0,
        },
      });
    }

    // Fetch all experiments
    const result = await qdrantRequest(
      `/collections/${EXPERIMENTS_COLLECTION}/points/scroll`,
      "POST",
      {
        limit: 100,
        with_payload: true,
      }
    );

    const experiments: Experiment[] = (result.result?.points || [])
      .map((p: { payload: Experiment }) => p.payload)
      .sort((a: Experiment, b: Experiment) =>
        (b.started_at || "").localeCompare(a.started_at || "")
      );

    // Build summary
    const summary = {
      total: experiments.length,
      completed: experiments.filter((e) => e.status === "completed").length,
      failed: experiments.filter((e) => e.status === "failed").length,
      running: experiments.filter((e) => e.status === "running").length,
    };

    // Build comparison data
    const comparison = experiments
      .filter((e) => e.status === "completed")
      .map((e) => ({
        id: e.experiment_id,
        name: e.config?.name || "Unknown",
        provider: e.config?.embedding_provider || "unknown",
        model: e.config?.embedding_model || "unknown",
        dimensions: e.config?.dimensions || 0,
        mrl: e.config?.use_mrl || false,
        quantized: e.config?.use_quantization || false,
        sparse: e.config?.use_sparse || false,
        vectors: e.total_vectors,
        throughput: e.vectors_per_second,
        embedTimePerVec: e.avg_embedding_time_per_vector_ms,
        searchP50: e.search_latency_p50_ms,
        searchP95: e.search_latency_p95_ms,
        sizeMB: e.estimated_total_size_mb,
        startedAt: e.started_at,
      }));

    return NextResponse.json({
      experiments,
      summary,
      comparison,
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error("Error fetching experiments:", error);
    return NextResponse.json(
      { error: "Failed to fetch experiments", details: String(error) },
      { status: 500 }
    );
  }
}
