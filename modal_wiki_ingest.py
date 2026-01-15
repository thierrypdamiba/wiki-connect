"""
Modal App for Wikipedia Parquet Ingestion with Multiple Configs

GPU-accelerated embedding with parallel processing.
Each config runs on a separate parquet file for comparison.

Usage:
    # Single config
    modal run modal_wiki_ingest.py::ingest_parquet --config openai_256_mrl --parquet-url s3://...

    # Multiple configs in parallel (each on different parquet)
    modal run modal_wiki_ingest.py::run_experiment_suite

    # View results
    modal run modal_wiki_ingest.py::show_results
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import json

import modal

# =============================================================================
# Modal App Configuration
# =============================================================================

app = modal.App("wiki-parquet-ingest")

# Volumes for data and results
data_volume = modal.Volume.from_name("wiki-parquet-data", create_if_missing=True)
results_volume = modal.Volume.from_name("wiki-ingest-results", create_if_missing=True)

# Base image with core dependencies
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Qdrant
        "qdrant-client>=1.12.0",
        # Embeddings
        "openai>=1.50.0",
        "cohere>=5.0.0",
        # Data processing
        "pyarrow>=15.0.0",
        "pandas>=2.2.0",
        # Utils
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
    )
)

# GPU image with FastEmbed
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "qdrant-client>=1.12.0",
        "fastembed-gpu>=0.4.0",
        "pyarrow>=15.0.0",
        "pandas>=2.2.0",
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
        "torch>=2.2.0",
    )
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class IngestConfig:
    """Configuration for a single ingestion run"""
    name: str
    embedding_provider: str  # openai, cohere, fastembed
    embedding_model: str
    dimensions: int
    use_mrl: bool = False
    use_quantization: bool = False
    quantization_type: Optional[str] = None
    use_sparse: bool = False
    batch_size: int = 1000
    defer_indexing: bool = True
    on_disk: bool = True
    shard_number: int = 2


@dataclass
class IngestMetrics:
    """Metrics from ingestion run"""
    experiment_id: str
    config_name: str
    config: Dict[str, Any]
    parquet_file: str
    collection_name: str
    started_at: str
    completed_at: Optional[str] = None
    status: str = "running"
    total_rows: int = 0
    total_vectors: int = 0
    total_batches: int = 0
    embedding_time_ms: float = 0
    upsert_time_ms: float = 0
    total_time_ms: float = 0
    vectors_per_second: float = 0
    search_latency_p50_ms: Optional[float] = None
    search_latency_p95_ms: Optional[float] = None
    estimated_size_mb: float = 0
    errors: List[str] = field(default_factory=list)


# Predefined configurations
CONFIGS: Dict[str, IngestConfig] = {
    "openai_256_mrl": IngestConfig(
        name="OpenAI 256d MRL",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        dimensions=256,
        use_mrl=True,
        batch_size=2000,
    ),
    "openai_512_mrl": IngestConfig(
        name="OpenAI 512d MRL",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        dimensions=512,
        use_mrl=True,
        batch_size=1500,
    ),
    "openai_1536": IngestConfig(
        name="OpenAI 1536d",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        dimensions=1536,
        use_mrl=False,
        batch_size=500,
    ),
    "openai_256_quantized": IngestConfig(
        name="OpenAI 256d INT8",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        dimensions=256,
        use_mrl=True,
        use_quantization=True,
        quantization_type="int8",
        batch_size=2000,
    ),
    "cohere_256_mrl": IngestConfig(
        name="Cohere 256d MRL",
        embedding_provider="cohere",
        embedding_model="embed-english-v3.0",
        dimensions=256,
        use_mrl=True,
        batch_size=96,
    ),
    "cohere_1024": IngestConfig(
        name="Cohere 1024d",
        embedding_provider="cohere",
        embedding_model="embed-english-v3.0",
        dimensions=1024,
        use_mrl=False,
        batch_size=96,
    ),
    "fastembed_bge_384": IngestConfig(
        name="FastEmbed BGE-small 384d",
        embedding_provider="fastembed",
        embedding_model="BAAI/bge-small-en-v1.5",
        dimensions=384,
        batch_size=500,
    ),
    "fastembed_bge_768": IngestConfig(
        name="FastEmbed BGE-base 768d",
        embedding_provider="fastembed",
        embedding_model="BAAI/bge-base-en-v1.5",
        dimensions=768,
        batch_size=200,
    ),
}


# =============================================================================
# Embedding Functions
# =============================================================================

def embed_openai(texts: List[str], model: str, dimensions: int, use_mrl: bool, api_key: str) -> List[List[float]]:
    """Embed with OpenAI"""
    import openai
    client = openai.OpenAI(api_key=api_key)

    kwargs = {"model": model, "input": texts}
    if use_mrl:
        kwargs["dimensions"] = dimensions

    response = client.embeddings.create(**kwargs)
    return [d.embedding for d in response.data]


def embed_cohere(texts: List[str], model: str, dimensions: int, use_mrl: bool, api_key: str) -> List[List[float]]:
    """Embed with Cohere"""
    import cohere
    client = cohere.ClientV2(api_key=api_key)

    kwargs = {"model": model, "texts": texts, "input_type": "search_document"}
    if use_mrl:
        kwargs["output_dimension"] = dimensions

    response = client.embed(**kwargs)
    return [list(e) for e in response.embeddings]


def embed_fastembed(texts: List[str], model: str, embedder=None) -> List[List[float]]:
    """Embed with FastEmbed (GPU)"""
    from fastembed import TextEmbedding
    if embedder is None:
        embedder = TextEmbedding(model)
    return [list(e) for e in embedder.embed(texts)]


# =============================================================================
# Main Ingestion Function (CPU - for OpenAI/Cohere)
# =============================================================================

@app.function(
    image=base_image,
    volumes={
        "/data": data_volume,
        "/results": results_volume,
    },
    timeout=3600 * 4,  # 4 hours
    secrets=[
        modal.Secret.from_name("qdrant"),
        modal.Secret.from_name("openai", required_keys=["OPENAI_API_KEY"]),
        modal.Secret.from_name("cohere", required_keys=["COHERE_API_KEY"]),
    ],
    cpu=4,
    memory=8192,
)
def ingest_parquet_cpu(
    config_name: str,
    parquet_path: str,
    text_column: str = "text",
    collection_suffix: str = "",
) -> Dict:
    """
    Ingest parquet file with OpenAI or Cohere embeddings.

    Args:
        config_name: Name of config from CONFIGS
        parquet_path: Path to parquet file (in /data volume or URL)
        text_column: Column containing text to embed
        collection_suffix: Optional suffix for collection name
    """
    import time
    import uuid
    import pyarrow.parquet as pq
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        ScalarQuantization, ScalarType, HnswConfigDiff,
    )

    config = CONFIGS[config_name]
    experiment_id = uuid.uuid4().hex[:12]
    collection_name = f"wiki_{config_name}_{experiment_id}"
    if collection_suffix:
        collection_name += f"_{collection_suffix}"

    metrics = IngestMetrics(
        experiment_id=experiment_id,
        config_name=config.name,
        config=asdict(config),
        parquet_file=parquet_path,
        collection_name=collection_name,
        started_at=datetime.utcnow().isoformat(),
    )

    # Get secrets
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_key = os.environ.get("QDRANT_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    cohere_key = os.environ.get("COHERE_API_KEY")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=300)

    total_start = time.time()

    try:
        # Create collection
        vectors_config = VectorParams(
            size=config.dimensions,
            distance=Distance.COSINE,
            on_disk=config.on_disk,
        )

        quantization = None
        if config.use_quantization and config.quantization_type == "int8":
            quantization = ScalarQuantization(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )

        hnsw_config = HnswConfigDiff(m=0) if config.defer_indexing else None

        try:
            client.delete_collection(collection_name)
        except:
            pass

        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            quantization_config=quantization,
            hnsw_config=hnsw_config,
            shard_number=config.shard_number,
        )

        print(f"[{config.name}] Created collection: {collection_name}")

        # Read parquet
        parquet_file = pq.ParquetFile(parquet_path)
        metrics.total_rows = parquet_file.metadata.num_rows

        total_embedding_time = 0
        total_upsert_time = 0
        batch_count = 0
        vector_count = 0

        for batch in parquet_file.iter_batches(batch_size=config.batch_size, columns=[text_column]):
            texts = [str(t) for t in batch[text_column].to_pylist() if t]
            if not texts:
                continue

            batch_count += 1

            # Embed
            embed_start = time.time()
            if config.embedding_provider == "openai":
                vectors = embed_openai(texts, config.embedding_model, config.dimensions, config.use_mrl, openai_key)
            else:
                vectors = embed_cohere(texts, config.embedding_model, config.dimensions, config.use_mrl, cohere_key)
            embed_time = (time.time() - embed_start) * 1000
            total_embedding_time += embed_time

            # Create points
            points = [
                PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vec,
                    payload={"text": text[:1000], "batch": batch_count},
                )
                for text, vec in zip(texts, vectors)
            ]

            # Upsert
            upsert_start = time.time()
            client.upsert(collection_name=collection_name, points=points, wait=True)
            upsert_time = (time.time() - upsert_start) * 1000
            total_upsert_time += upsert_time

            vector_count += len(points)

            if batch_count % 10 == 0:
                print(f"  [{config.name}] Batch {batch_count}: {vector_count:,} vectors - "
                      f"Embed: {embed_time:.0f}ms, Upsert: {upsert_time:.0f}ms")

        # Re-enable indexing
        if config.defer_indexing and vector_count > 0:
            print(f"  [{config.name}] Re-enabling HNSW...")
            index_start = time.time()
            client.update_collection(
                collection_name=collection_name,
                hnsw_config=HnswConfigDiff(m=16),
            )
            while True:
                info = client.get_collection(collection_name)
                if info.status.value == "green":
                    break
                time.sleep(1)
            print(f"  [{config.name}] Indexing done in {(time.time() - index_start):.1f}s")

        total_time = (time.time() - total_start) * 1000

        # Metrics
        metrics.total_vectors = vector_count
        metrics.total_batches = batch_count
        metrics.embedding_time_ms = total_embedding_time
        metrics.upsert_time_ms = total_upsert_time
        metrics.total_time_ms = total_time
        metrics.vectors_per_second = vector_count / (total_time / 1000) if total_time > 0 else 0

        bytes_per_float = 1 if config.use_quantization else 4
        metrics.estimated_size_mb = (config.dimensions * bytes_per_float * vector_count) / (1024 * 1024)

        # Search test
        if vector_count > 0:
            test_result = client.scroll(collection_name=collection_name, limit=5, with_payload=True)
            test_texts = [p.payload.get("text", "")[:200] for p in test_result[0]]

            if test_texts and config.embedding_provider == "openai":
                test_vecs = embed_openai(test_texts, config.embedding_model, config.dimensions, config.use_mrl, openai_key)
            elif test_texts and config.embedding_provider == "cohere":
                test_vecs = embed_cohere(test_texts, config.embedding_model, config.dimensions, config.use_mrl, cohere_key)
            else:
                test_vecs = []

            latencies = []
            for vec in test_vecs:
                start = time.time()
                client.query_points(collection_name=collection_name, query=vec, limit=10)
                latencies.append((time.time() - start) * 1000)

            if latencies:
                latencies.sort()
                metrics.search_latency_p50_ms = latencies[len(latencies) // 2]
                metrics.search_latency_p95_ms = latencies[int(len(latencies) * 0.95)]

        metrics.status = "completed"
        metrics.completed_at = datetime.utcnow().isoformat()

        print(f"\n[{config.name}] COMPLETED:")
        print(f"  Vectors: {metrics.total_vectors:,}")
        print(f"  Throughput: {metrics.vectors_per_second:.1f} vec/s")
        print(f"  Search P50: {metrics.search_latency_p50_ms}ms")

    except Exception as e:
        metrics.status = "failed"
        metrics.errors.append(str(e))
        metrics.completed_at = datetime.utcnow().isoformat()
        print(f"[{config.name}] FAILED: {e}")

    # Save results
    result_path = f"/results/{experiment_id}.json"
    with open(result_path, "w") as f:
        json.dump(asdict(metrics), f, indent=2)

    return asdict(metrics)


# =============================================================================
# GPU Ingestion Function (for FastEmbed)
# =============================================================================

@app.function(
    image=gpu_image,
    gpu="T4",
    volumes={
        "/data": data_volume,
        "/results": results_volume,
    },
    timeout=3600 * 4,
    secrets=[modal.Secret.from_name("qdrant")],
)
def ingest_parquet_gpu(
    config_name: str,
    parquet_path: str,
    text_column: str = "text",
    collection_suffix: str = "",
) -> Dict:
    """
    Ingest parquet file with GPU-accelerated FastEmbed.
    """
    import time
    import uuid
    import pyarrow.parquet as pq
    from fastembed import TextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, HnswConfigDiff,
    )

    config = CONFIGS[config_name]
    experiment_id = uuid.uuid4().hex[:12]
    collection_name = f"wiki_{config_name}_{experiment_id}"

    metrics = IngestMetrics(
        experiment_id=experiment_id,
        config_name=config.name,
        config=asdict(config),
        parquet_file=parquet_path,
        collection_name=collection_name,
        started_at=datetime.utcnow().isoformat(),
    )

    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_key = os.environ.get("QDRANT_API_KEY")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=300)
    embedder = TextEmbedding(config.embedding_model)

    total_start = time.time()

    try:
        # Create collection
        vectors_config = VectorParams(
            size=config.dimensions,
            distance=Distance.COSINE,
            on_disk=config.on_disk,
        )

        hnsw_config = HnswConfigDiff(m=0) if config.defer_indexing else None

        try:
            client.delete_collection(collection_name)
        except:
            pass

        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            hnsw_config=hnsw_config,
            shard_number=config.shard_number,
        )

        print(f"[{config.name}] Created collection: {collection_name}")

        # Read and process
        parquet_file = pq.ParquetFile(parquet_path)
        metrics.total_rows = parquet_file.metadata.num_rows

        total_embedding_time = 0
        total_upsert_time = 0
        batch_count = 0
        vector_count = 0

        for batch in parquet_file.iter_batches(batch_size=config.batch_size, columns=[text_column]):
            texts = [str(t) for t in batch[text_column].to_pylist() if t]
            if not texts:
                continue

            batch_count += 1

            # Embed on GPU
            embed_start = time.time()
            vectors = [list(e) for e in embedder.embed(texts)]
            embed_time = (time.time() - embed_start) * 1000
            total_embedding_time += embed_time

            # Create points
            points = [
                PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vec,
                    payload={"text": text[:1000], "batch": batch_count},
                )
                for text, vec in zip(texts, vectors)
            ]

            # Upsert
            upsert_start = time.time()
            client.upsert(collection_name=collection_name, points=points, wait=True)
            upsert_time = (time.time() - upsert_start) * 1000
            total_upsert_time += upsert_time

            vector_count += len(points)

            if batch_count % 10 == 0:
                print(f"  [{config.name}] Batch {batch_count}: {vector_count:,} vectors")

        # Re-enable indexing
        if config.defer_indexing and vector_count > 0:
            print(f"  [{config.name}] Re-enabling HNSW...")
            client.update_collection(
                collection_name=collection_name,
                hnsw_config=HnswConfigDiff(m=16),
            )
            while True:
                info = client.get_collection(collection_name)
                if info.status.value == "green":
                    break
                time.sleep(1)

        total_time = (time.time() - total_start) * 1000

        metrics.total_vectors = vector_count
        metrics.total_batches = batch_count
        metrics.embedding_time_ms = total_embedding_time
        metrics.upsert_time_ms = total_upsert_time
        metrics.total_time_ms = total_time
        metrics.vectors_per_second = vector_count / (total_time / 1000) if total_time > 0 else 0
        metrics.estimated_size_mb = (config.dimensions * 4 * vector_count) / (1024 * 1024)
        metrics.status = "completed"
        metrics.completed_at = datetime.utcnow().isoformat()

        print(f"\n[{config.name}] COMPLETED: {metrics.total_vectors:,} vectors @ {metrics.vectors_per_second:.1f}/s")

    except Exception as e:
        metrics.status = "failed"
        metrics.errors.append(str(e))
        metrics.completed_at = datetime.utcnow().isoformat()
        print(f"[{config.name}] FAILED: {e}")

    # Save results
    result_path = f"/results/{experiment_id}.json"
    with open(result_path, "w") as f:
        json.dump(asdict(metrics), f, indent=2)

    return asdict(metrics)


# =============================================================================
# Experiment Suite
# =============================================================================

@app.function(
    image=base_image,
    volumes={"/data": data_volume, "/results": results_volume},
    timeout=3600,
)
def run_experiment_suite(
    parquet_files: List[str],
    configs: List[str] = None,
) -> List[Dict]:
    """
    Run multiple configs on different parquet files in parallel.

    Args:
        parquet_files: List of parquet file paths (one per config)
        configs: Config names to run (defaults to all)
    """
    if configs is None:
        configs = list(CONFIGS.keys())

    # Match configs to parquet files
    jobs = []
    for i, config_name in enumerate(configs):
        parquet_idx = i % len(parquet_files)
        parquet_path = parquet_files[parquet_idx]

        config = CONFIGS[config_name]
        if config.embedding_provider == "fastembed":
            jobs.append(ingest_parquet_gpu.spawn(config_name, parquet_path))
        else:
            jobs.append(ingest_parquet_cpu.spawn(config_name, parquet_path))

    # Collect results
    results = []
    for job in jobs:
        try:
            result = job.get()
            results.append(result)
        except Exception as e:
            results.append({"error": str(e)})

    return results


@app.function(
    image=base_image,
    volumes={"/results": results_volume},
)
def show_results() -> str:
    """Show all experiment results"""
    import os
    import json

    results = []
    for filename in os.listdir("/results"):
        if filename.endswith(".json"):
            with open(f"/results/{filename}") as f:
                results.append(json.load(f))

    if not results:
        return "No results found."

    # Sort by started_at
    results.sort(key=lambda x: x.get("started_at", ""), reverse=True)

    output = ["=" * 80, "EXPERIMENT RESULTS", "=" * 80, ""]

    headers = ["Config", "Vectors", "Vec/s", "Embed ms/v", "P50 ms", "Size MB", "Status"]
    output.append("  ".join(f"{h:<15}" for h in headers))
    output.append("-" * 110)

    for r in results:
        row = [
            r.get("config_name", "?")[:13],
            f"{r.get('total_vectors', 0):,}",
            f"{r.get('vectors_per_second', 0):.1f}",
            f"{r.get('embedding_time_ms', 0) / max(r.get('total_vectors', 1), 1):.2f}",
            str(r.get("search_latency_p50_ms", "N/A")),
            f"{r.get('estimated_size_mb', 0):.2f}",
            r.get("status", "?"),
        ]
        output.append("  ".join(f"{v:<15}" for v in row))

    return "\n".join(output)


# =============================================================================
# Local Entry Point
# =============================================================================

@app.local_entrypoint()
def main():
    """List available configs"""
    print("\nAvailable Configurations:")
    print("=" * 60)
    for name, config in CONFIGS.items():
        print(f"\n{name}:")
        print(f"  {config.name}")
        print(f"  Provider: {config.embedding_provider}")
        print(f"  Dimensions: {config.dimensions}, MRL: {config.use_mrl}")

    print("\n\nUsage:")
    print("  modal run modal_wiki_ingest.py::ingest_parquet_cpu --config-name openai_256_mrl --parquet-path /data/wiki.parquet")
    print("  modal run modal_wiki_ingest.py::show_results")
