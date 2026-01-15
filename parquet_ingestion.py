"""
Parquet-Based Large Scale Ingestion for Qdrant

Based on Qdrant best practices:
- https://qdrant.tech/documentation/database-tutorials/bulk-upload/
- https://qdrant.tech/course/essentials/day-4/large-scale-ingestion/

Each config processes a single parquet file and tracks all metrics.
Run multiple configs in parallel to compare performance.

Usage:
    # Run single config on a parquet file
    python parquet_ingestion.py --parquet data.parquet --config openai_256_mrl

    # Run multiple configs in parallel on different parquets
    python parquet_ingestion.py --parquet-dir ./parquets --configs openai_256_mrl cohere_256_mrl --parallel 2

    # List available configs
    python parquet_ingestion.py --list-configs

    # View experiment results
    python parquet_ingestion.py --results
"""

import os
import sys
import time
import uuid
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Generator, Tuple
from dataclasses import dataclass, asdict, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

# Try to import optional dependencies
try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVectorParams,
    ScalarQuantization,
    ScalarType,
    HnswConfigDiff,
    OptimizersConfigDiff,
)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

EXPERIMENTS_COLLECTION = "ingestion_experiments"


@dataclass
class IngestionConfig:
    """Configuration for a parquet ingestion run"""
    name: str
    embedding_provider: str  # openai, cohere, fastembed
    embedding_model: str
    dimensions: int

    # MRL (Matryoshka) support
    use_mrl: bool = False

    # Quantization
    use_quantization: bool = False
    quantization_type: Optional[str] = None  # int8, binary

    # Sparse vectors
    use_sparse: bool = False
    sparse_model: Optional[str] = None

    # Batch settings
    batch_size: int = 1000  # 1000-10000 recommended for large scale

    # Qdrant optimization settings (from bulk upload docs)
    defer_indexing: bool = True  # m=0 during upload, then re-enable
    on_disk: bool = True  # Store vectors on disk for large scale
    parallel_upload: int = 4  # Number of parallel upload threads
    shard_number: int = 2  # 2-4 shards per machine recommended

    # Parquet reading
    parquet_row_group_size: int = 10000  # Rows to read at once
    text_column: str = "text"  # Column containing text to embed


@dataclass
class IngestionMetrics:
    """Detailed metrics for ingestion run"""
    experiment_id: str
    config_name: str
    config: Dict[str, Any]
    parquet_file: str
    collection_name: str

    started_at: str
    completed_at: Optional[str] = None
    status: str = "running"

    # Volume
    total_rows: int = 0
    total_vectors: int = 0
    total_batches: int = 0
    failed_batches: int = 0

    # Timing (ms)
    parquet_read_time_ms: float = 0
    embedding_time_ms: float = 0
    upsert_time_ms: float = 0
    total_time_ms: float = 0

    # Per-operation
    avg_embedding_time_per_vector_ms: float = 0
    avg_upsert_time_per_batch_ms: float = 0
    avg_batch_time_ms: float = 0

    # Throughput
    vectors_per_second: float = 0
    mb_per_second: float = 0

    # Memory/Size
    parquet_size_mb: float = 0
    estimated_vector_size_bytes: int = 0
    estimated_total_vectors_mb: float = 0

    # Search quality (post-ingestion test)
    search_latency_p50_ms: Optional[float] = None
    search_latency_p95_ms: Optional[float] = None
    search_latency_p99_ms: Optional[float] = None

    # Index stats (after re-enabling indexing)
    indexing_time_ms: Optional[float] = None
    final_segment_count: Optional[int] = None

    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


# Predefined configurations optimized for large-scale parquet ingestion
CONFIGS: Dict[str, IngestionConfig] = {
    # OpenAI configs
    "openai_256_mrl_fast": IngestionConfig(
        name="OpenAI 256d MRL - Speed Optimized",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        dimensions=256,
        use_mrl=True,
        batch_size=2000,
        defer_indexing=True,
        on_disk=True,
        parallel_upload=8,
    ),
    "openai_256_mrl_balanced": IngestionConfig(
        name="OpenAI 256d MRL - Balanced",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        dimensions=256,
        use_mrl=True,
        batch_size=1000,
        defer_indexing=True,
        on_disk=True,
        parallel_upload=4,
    ),
    "openai_512_mrl": IngestionConfig(
        name="OpenAI 512d MRL",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        dimensions=512,
        use_mrl=True,
        batch_size=1000,
        defer_indexing=True,
        on_disk=True,
    ),
    "openai_1536_full": IngestionConfig(
        name="OpenAI 1536d Full",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        dimensions=1536,
        use_mrl=False,
        batch_size=500,  # Smaller batches for larger vectors
        defer_indexing=True,
        on_disk=True,
    ),
    "openai_3072_large": IngestionConfig(
        name="OpenAI 3072d Large Model",
        embedding_provider="openai",
        embedding_model="text-embedding-3-large",
        dimensions=3072,
        use_mrl=False,
        batch_size=200,  # Even smaller for large model
        defer_indexing=True,
        on_disk=True,
    ),

    # Quantized configs
    "openai_256_quantized_int8": IngestionConfig(
        name="OpenAI 256d + INT8 Quantization",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        dimensions=256,
        use_mrl=True,
        use_quantization=True,
        quantization_type="int8",
        batch_size=2000,
        defer_indexing=True,
        on_disk=True,
    ),
    "openai_512_quantized_int8": IngestionConfig(
        name="OpenAI 512d + INT8 Quantization",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        dimensions=512,
        use_mrl=True,
        use_quantization=True,
        quantization_type="int8",
        batch_size=1500,
        defer_indexing=True,
        on_disk=True,
    ),

    # Hybrid configs (Dense + Sparse)
    "openai_256_hybrid_bm25": IngestionConfig(
        name="OpenAI 256d + BM25 Sparse",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        dimensions=256,
        use_mrl=True,
        use_sparse=True,
        sparse_model="bm25",
        batch_size=1000,
        defer_indexing=True,
        on_disk=True,
    ),

    # Cohere configs
    "cohere_256_mrl": IngestionConfig(
        name="Cohere embed-v3 256d MRL",
        embedding_provider="cohere",
        embedding_model="embed-english-v3.0",
        dimensions=256,
        use_mrl=True,
        batch_size=96,  # Cohere batch limit
        defer_indexing=True,
        on_disk=True,
    ),
    "cohere_1024": IngestionConfig(
        name="Cohere embed-v3 1024d",
        embedding_provider="cohere",
        embedding_model="embed-english-v3.0",
        dimensions=1024,
        use_mrl=False,
        batch_size=96,
        defer_indexing=True,
        on_disk=True,
    ),

    # FastEmbed (local, no API costs)
    "fastembed_bge_384": IngestionConfig(
        name="FastEmbed BGE-small 384d (Local)",
        embedding_provider="fastembed",
        embedding_model="BAAI/bge-small-en-v1.5",
        dimensions=384,
        use_mrl=False,
        batch_size=500,
        defer_indexing=True,
        on_disk=True,
    ),
    "fastembed_bge_768": IngestionConfig(
        name="FastEmbed BGE-base 768d (Local)",
        embedding_provider="fastembed",
        embedding_model="BAAI/bge-base-en-v1.5",
        dimensions=768,
        use_mrl=False,
        batch_size=200,
        defer_indexing=True,
        on_disk=True,
    ),

    # Memory-optimized (no defer, for smaller datasets)
    "openai_256_no_defer": IngestionConfig(
        name="OpenAI 256d - Index During Upload",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        dimensions=256,
        use_mrl=True,
        batch_size=1000,
        defer_indexing=False,  # Index as we go
        on_disk=True,
    ),
}


class Embedder:
    """Unified embedder for different providers"""

    def __init__(self, config: IngestionConfig):
        self.config = config
        self._client = None

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self.config.embedding_provider == "openai":
            return self._embed_openai(texts)
        elif self.config.embedding_provider == "cohere":
            return self._embed_cohere(texts)
        elif self.config.embedding_provider == "fastembed":
            return self._embed_fastembed(texts)
        else:
            raise ValueError(f"Unknown provider: {self.config.embedding_provider}")

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        import openai
        if not self._client:
            self._client = openai.OpenAI(api_key=OPENAI_API_KEY)

        kwargs = {"model": self.config.embedding_model, "input": texts}
        if self.config.use_mrl:
            kwargs["dimensions"] = self.config.dimensions

        response = self._client.embeddings.create(**kwargs)
        return [d.embedding for d in response.data]

    def _embed_cohere(self, texts: List[str]) -> List[List[float]]:
        import cohere
        if not self._client:
            self._client = cohere.ClientV2(api_key=COHERE_API_KEY)

        kwargs = {
            "model": self.config.embedding_model,
            "texts": texts,
            "input_type": "search_document",
        }
        if self.config.use_mrl:
            kwargs["output_dimension"] = self.config.dimensions

        response = self._client.embed(**kwargs)
        return [list(e) for e in response.embeddings]

    def _embed_fastembed(self, texts: List[str]) -> List[List[float]]:
        from fastembed import TextEmbedding
        if not self._client:
            self._client = TextEmbedding(self.config.embedding_model)
        return [list(e) for e in self._client.embed(texts)]


def generate_bm25_sparse(text: str) -> Dict[str, List]:
    """Generate BM25 sparse vector"""
    tokens = text.lower().split()
    tokens = [t for t in tokens if len(t) > 2 and t.isalnum()]

    term_freq: Dict[str, int] = {}
    for token in tokens:
        term_freq[token] = term_freq.get(token, 0) + 1

    indices, values = [], []
    for token, freq in term_freq.items():
        h = sum(ord(c) << (i % 8) for i, c in enumerate(token)) % 30000
        tf = freq / (freq + 1.2)
        indices.append(h)
        values.append(tf)

    return {"indices": indices, "values": values}


def read_parquet_batches(
    parquet_path: Path,
    text_column: str,
    batch_size: int,
) -> Generator[Tuple[List[str], int], None, None]:
    """Stream parquet file in batches"""
    if HAS_PYARROW:
        parquet_file = pq.ParquetFile(parquet_path)
        total_rows = parquet_file.metadata.num_rows

        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=[text_column]):
            texts = batch[text_column].to_pylist()
            texts = [str(t) for t in texts if t]  # Filter empty
            yield texts, total_rows

    elif HAS_PANDAS:
        # Fall back to pandas chunked reading
        total_rows = 0
        for chunk in pd.read_parquet(parquet_path, columns=[text_column], chunksize=batch_size):
            if total_rows == 0:
                # Estimate total from file size
                total_rows = len(chunk) * 100  # Rough estimate
            texts = chunk[text_column].dropna().astype(str).tolist()
            yield texts, total_rows
    else:
        raise ImportError("Neither pyarrow nor pandas installed. Run: pip install pyarrow")


def run_ingestion(
    config: IngestionConfig,
    parquet_path: Path,
    collection_suffix: str = "",
) -> IngestionMetrics:
    """Run a single ingestion experiment"""

    experiment_id = uuid.uuid4().hex[:12]
    collection_name = f"exp_{config.name.lower().replace(' ', '_').replace('-', '_')[:30]}_{experiment_id}"

    metrics = IngestionMetrics(
        experiment_id=experiment_id,
        config_name=config.name,
        config=asdict(config),
        parquet_file=str(parquet_path),
        collection_name=collection_name,
        started_at=datetime.utcnow().isoformat(),
        parquet_size_mb=parquet_path.stat().st_size / (1024 * 1024) if parquet_path.exists() else 0,
    )

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=300)
    embedder = Embedder(config)

    total_start = time.time()

    try:
        # Create collection with optimized settings
        vectors_config = VectorParams(
            size=config.dimensions,
            distance=Distance.COSINE,
            on_disk=config.on_disk,
        )

        sparse_config = None
        if config.use_sparse:
            sparse_config = {"sparse": SparseVectorParams()}

        quantization = None
        if config.use_quantization and config.quantization_type == "int8":
            quantization = ScalarQuantization(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )

        # HNSW config - defer indexing for bulk upload
        hnsw_config = None
        if config.defer_indexing:
            hnsw_config = HnswConfigDiff(m=0)  # Disable HNSW during upload

        # Delete existing
        try:
            client.delete_collection(collection_name)
        except:
            pass

        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
            quantization_config=quantization,
            hnsw_config=hnsw_config,
            shard_number=config.shard_number,
        )

        print(f"\n[{config.name}] Created collection: {collection_name}")
        print(f"  Settings: on_disk={config.on_disk}, defer_indexing={config.defer_indexing}, shards={config.shard_number}")

        # Process parquet in batches
        total_embedding_time = 0
        total_upsert_time = 0
        total_read_time = 0
        batch_count = 0
        vector_count = 0

        for texts, total_rows in read_parquet_batches(
            parquet_path, config.text_column, config.batch_size
        ):
            if metrics.total_rows == 0:
                metrics.total_rows = total_rows

            if not texts:
                continue

            batch_count += 1

            # Time: Read (already done in generator)
            read_start = time.time()
            # Reading happened in generator
            read_time = 0  # Can't measure separately

            # Time: Embedding
            embed_start = time.time()
            try:
                vectors = embedder.embed_batch(texts)
            except Exception as e:
                metrics.errors.append(f"Batch {batch_count} embed error: {str(e)[:100]}")
                metrics.failed_batches += 1
                continue
            embed_time = (time.time() - embed_start) * 1000
            total_embedding_time += embed_time

            # Create points
            points = []
            for i, (text, vector) in enumerate(zip(texts, vectors)):
                point_id = uuid.uuid4().hex
                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": text[:1000],
                        "batch": batch_count,
                    },
                )

                if config.use_sparse:
                    sparse = generate_bm25_sparse(text)
                    point.vector = {
                        "dense": vector,
                        "sparse": sparse,
                    }

                points.append(point)

            # Time: Upsert
            upsert_start = time.time()
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,
                )
            except Exception as e:
                metrics.errors.append(f"Batch {batch_count} upsert error: {str(e)[:100]}")
                metrics.failed_batches += 1
                continue
            upsert_time = (time.time() - upsert_start) * 1000
            total_upsert_time += upsert_time

            vector_count += len(points)

            # Progress
            if batch_count % 10 == 0 or batch_count == 1:
                progress = (vector_count / total_rows * 100) if total_rows > 0 else 0
                print(f"  [{config.name}] Batch {batch_count}: {vector_count:,} vectors "
                      f"({progress:.1f}%) - Embed: {embed_time:.0f}ms, Upsert: {upsert_time:.0f}ms")

        # Re-enable indexing if deferred
        indexing_start = None
        if config.defer_indexing and vector_count > 0:
            print(f"  [{config.name}] Re-enabling HNSW indexing (m=16)...")
            indexing_start = time.time()
            client.update_collection(
                collection_name=collection_name,
                hnsw_config=HnswConfigDiff(m=16),
            )
            # Wait for indexing to complete
            while True:
                info = client.get_collection(collection_name)
                if info.status.value == "green":
                    break
                time.sleep(1)
            metrics.indexing_time_ms = (time.time() - indexing_start) * 1000
            print(f"  [{config.name}] Indexing complete in {metrics.indexing_time_ms:.0f}ms")

        total_time = (time.time() - total_start) * 1000

        # Calculate metrics
        metrics.total_vectors = vector_count
        metrics.total_batches = batch_count
        metrics.embedding_time_ms = total_embedding_time
        metrics.upsert_time_ms = total_upsert_time
        metrics.total_time_ms = total_time

        if batch_count > 0:
            metrics.avg_upsert_time_per_batch_ms = total_upsert_time / batch_count
            metrics.avg_batch_time_ms = total_time / batch_count

        if vector_count > 0:
            metrics.avg_embedding_time_per_vector_ms = total_embedding_time / vector_count
            metrics.vectors_per_second = vector_count / (total_time / 1000)

        # Size calculations
        bytes_per_float = 4
        if config.use_quantization and config.quantization_type == "int8":
            bytes_per_float = 1

        metrics.estimated_vector_size_bytes = config.dimensions * bytes_per_float
        metrics.estimated_total_vectors_mb = (
            metrics.estimated_vector_size_bytes * vector_count
        ) / (1024 * 1024)

        if total_time > 0:
            metrics.mb_per_second = metrics.estimated_total_vectors_mb / (total_time / 1000)

        # Search latency test
        if vector_count > 0:
            print(f"  [{config.name}] Running search latency test...")
            search_latencies = []

            # Get some test texts from the collection
            test_result = client.scroll(
                collection_name=collection_name,
                limit=10,
                with_payload=True,
            )
            test_texts = [p.payload.get("text", "")[:200] for p in test_result[0]]

            if test_texts:
                test_vectors = embedder.embed_batch(test_texts)
                for vec in test_vectors:
                    search_start = time.time()
                    client.query_points(
                        collection_name=collection_name,
                        query=vec,
                        limit=10,
                    )
                    search_latencies.append((time.time() - search_start) * 1000)

                search_latencies.sort()
                n = len(search_latencies)
                metrics.search_latency_p50_ms = search_latencies[n // 2]
                metrics.search_latency_p95_ms = search_latencies[int(n * 0.95)]
                metrics.search_latency_p99_ms = search_latencies[-1] if n > 0 else None

        # Get final segment count
        info = client.get_collection(collection_name)
        metrics.final_segment_count = info.segments_count

        metrics.status = "completed"
        metrics.completed_at = datetime.utcnow().isoformat()

        print(f"\n[{config.name}] COMPLETED:")
        print(f"  Vectors: {metrics.total_vectors:,}")
        print(f"  Total time: {metrics.total_time_ms / 1000:.1f}s")
        print(f"  Throughput: {metrics.vectors_per_second:.1f} vectors/sec")
        print(f"  Embed time/vec: {metrics.avg_embedding_time_per_vector_ms:.2f}ms")
        print(f"  Est. size: {metrics.estimated_total_vectors_mb:.2f}MB")
        print(f"  Search P50: {metrics.search_latency_p50_ms:.1f}ms" if metrics.search_latency_p50_ms else "  Search P50: N/A")

    except Exception as e:
        metrics.status = "failed"
        metrics.errors.append(str(e))
        metrics.completed_at = datetime.utcnow().isoformat()
        print(f"\n[{config.name}] FAILED: {e}")

    # Store results
    store_metrics(client, metrics)

    return metrics


def store_metrics(client: QdrantClient, metrics: IngestionMetrics):
    """Store experiment metrics in Qdrant"""
    try:
        client.get_collection(EXPERIMENTS_COLLECTION)
    except:
        client.create_collection(
            collection_name=EXPERIMENTS_COLLECTION,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),
        )

    point = PointStruct(
        id=metrics.experiment_id,
        vector=[0.0],
        payload=metrics.to_dict(),
    )
    client.upsert(collection_name=EXPERIMENTS_COLLECTION, points=[point])


def list_results(client: QdrantClient) -> List[Dict]:
    """List all experiment results"""
    try:
        results = client.scroll(
            collection_name=EXPERIMENTS_COLLECTION,
            limit=100,
            with_payload=True,
        )
        experiments = [p.payload for p in results[0]]
        return sorted(experiments, key=lambda x: x.get("started_at", ""), reverse=True)
    except:
        return []


def print_comparison_table(experiments: List[Dict]):
    """Print a comparison table of experiments"""
    if not experiments:
        print("No experiments found.")
        return

    print("\n" + "=" * 120)
    print("EXPERIMENT COMPARISON")
    print("=" * 120)

    headers = ["Config", "Dims", "MRL", "Quant", "Vectors", "Vec/s", "Embed ms/v", "P50 ms", "Size MB", "Status"]
    widths = [35, 6, 5, 6, 10, 10, 12, 10, 10, 10]

    print("".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))

    for exp in experiments:
        config = exp.get("config", {})
        row = [
            config.get("name", "?")[:33],
            str(config.get("dimensions", "?")),
            "Yes" if config.get("use_mrl") else "No",
            "INT8" if config.get("use_quantization") else "No",
            f"{exp.get('total_vectors', 0):,}",
            f"{exp.get('vectors_per_second', 0):.1f}",
            f"{exp.get('avg_embedding_time_per_vector_ms', 0):.2f}",
            f"{exp.get('search_latency_p50_ms', 'N/A')}",
            f"{exp.get('estimated_total_vectors_mb', 0):.2f}",
            exp.get("status", "?"),
        ]
        print("".join(str(v).ljust(w) for v, w in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(
        description="Parquet-based Qdrant ingestion with metrics tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--parquet", type=Path, help="Single parquet file to ingest")
    parser.add_argument("--parquet-dir", type=Path, help="Directory of parquet files")
    parser.add_argument("--config", help="Single config to run")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS.keys()), help="Multiple configs")
    parser.add_argument("--list-configs", action="store_true", help="List available configs")
    parser.add_argument("--results", action="store_true", help="Show experiment results")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel experiments")
    parser.add_argument("--text-column", default="text", help="Column name for text")

    args = parser.parse_args()

    if args.list_configs:
        print("\nAvailable Configurations:")
        print("=" * 70)
        for name, config in CONFIGS.items():
            print(f"\n{name}:")
            print(f"  {config.name}")
            print(f"  Provider: {config.embedding_provider}, Model: {config.embedding_model}")
            print(f"  Dimensions: {config.dimensions}, MRL: {config.use_mrl}")
            print(f"  Batch: {config.batch_size}, Shards: {config.shard_number}")
            if config.use_quantization:
                print(f"  Quantization: {config.quantization_type}")
            if config.use_sparse:
                print(f"  Sparse: {config.sparse_model}")
        return

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

    if args.results:
        experiments = list_results(client)
        print_comparison_table(experiments)
        return

    # Validate inputs
    if not args.parquet and not args.parquet_dir:
        print("Error: Must specify --parquet or --parquet-dir")
        parser.print_help()
        return

    if not args.config and not args.configs:
        print("Error: Must specify --config or --configs")
        print("\nAvailable configs:", ", ".join(CONFIGS.keys()))
        return

    # Get parquet files
    parquet_files = []
    if args.parquet:
        parquet_files = [args.parquet]
    elif args.parquet_dir:
        parquet_files = list(args.parquet_dir.glob("*.parquet"))

    if not parquet_files:
        print("Error: No parquet files found")
        return

    # Get configs
    configs_to_run = []
    if args.config:
        if args.config not in CONFIGS:
            print(f"Error: Unknown config '{args.config}'")
            return
        configs_to_run = [CONFIGS[args.config]]
    else:
        configs_to_run = [CONFIGS[name] for name in args.configs]

    # Override text column if specified
    for config in configs_to_run:
        config.text_column = args.text_column

    print(f"\nRunning {len(configs_to_run)} config(s) on {len(parquet_files)} parquet file(s)")

    # Run experiments
    all_results = []

    if args.parallel > 1 and len(configs_to_run) > 1:
        print(f"Running in parallel (max {args.parallel} workers)...")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {}
            for config in configs_to_run:
                for pq_file in parquet_files:
                    future = executor.submit(run_ingestion, config, pq_file)
                    futures[future] = (config.name, pq_file)

            for future in as_completed(futures):
                config_name, pq_file = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    print(f"Error in {config_name}: {e}")
    else:
        for config in configs_to_run:
            for pq_file in parquet_files:
                result = run_ingestion(config, pq_file)
                all_results.append(result)

    # Print summary
    print_comparison_table([r.to_dict() for r in all_results])


if __name__ == "__main__":
    main()
