"""
Ingestion Experiments - Compare different embedding configurations

Run multiple configurations in parallel and track all metrics for comparison.
Results are stored in Qdrant for later analysis.

Usage:
    python ingestion_experiments.py --configs openai_256 openai_512 cohere_256
    python ingestion_experiments.py --list  # Show all stored experiments
    python ingestion_experiments.py --compare exp1 exp2  # Compare two experiments
"""

import os
import sys
import json
import time
import uuid
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVectorParams,
    ScalarQuantization,
    ScalarType,
    Filter,
    FieldCondition,
    MatchValue,
)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

EXPERIMENTS_COLLECTION = "ingestion_experiments"


@dataclass
class ExperimentConfig:
    """Configuration for an embedding experiment"""
    name: str
    embedding_provider: str  # openai, cohere, fastembed
    embedding_model: str
    dimensions: int
    use_mrl: bool = False  # Matryoshka Representation Learning
    use_quantization: bool = False
    quantization_type: Optional[str] = None  # int8, binary
    use_sparse: bool = False
    sparse_model: Optional[str] = None
    batch_size: int = 100


@dataclass
class ExperimentMetrics:
    """Metrics tracked during ingestion"""
    experiment_id: str
    config: Dict[str, Any]
    started_at: str
    completed_at: Optional[str] = None
    status: str = "running"

    # Volume metrics
    total_texts: int = 0
    total_vectors: int = 0
    total_batches: int = 0

    # Timing metrics (all in milliseconds)
    total_embedding_time_ms: float = 0
    total_upsert_time_ms: float = 0
    total_time_ms: float = 0

    # Per-batch metrics
    avg_embedding_time_per_batch_ms: float = 0
    avg_upsert_time_per_batch_ms: float = 0
    avg_embedding_time_per_vector_ms: float = 0

    # Throughput
    vectors_per_second: float = 0

    # Memory/size metrics
    estimated_vector_size_bytes: int = 0
    estimated_total_size_mb: float = 0

    # Quality metrics (optional - from search tests)
    search_latency_p50_ms: Optional[float] = None
    search_latency_p95_ms: Optional[float] = None
    recall_at_10: Optional[float] = None

    # Errors
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


# Predefined configurations
CONFIGS: Dict[str, ExperimentConfig] = {
    "openai_256_mrl": ExperimentConfig(
        name="OpenAI text-embedding-3-small 256d MRL",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        dimensions=256,
        use_mrl=True,
        batch_size=100,
    ),
    "openai_512_mrl": ExperimentConfig(
        name="OpenAI text-embedding-3-small 512d MRL",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        dimensions=512,
        use_mrl=True,
        batch_size=100,
    ),
    "openai_1536": ExperimentConfig(
        name="OpenAI text-embedding-3-small 1536d",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        dimensions=1536,
        use_mrl=False,
        batch_size=100,
    ),
    "openai_3072_large": ExperimentConfig(
        name="OpenAI text-embedding-3-large 3072d",
        embedding_provider="openai",
        embedding_model="text-embedding-3-large",
        dimensions=3072,
        use_mrl=False,
        batch_size=50,  # Smaller batches for large model
    ),
    "cohere_256_mrl": ExperimentConfig(
        name="Cohere embed-v3 256d MRL",
        embedding_provider="cohere",
        embedding_model="embed-english-v3.0",
        dimensions=256,
        use_mrl=True,
        batch_size=96,  # Cohere limit
    ),
    "cohere_1024": ExperimentConfig(
        name="Cohere embed-v3 1024d",
        embedding_provider="cohere",
        embedding_model="embed-english-v3.0",
        dimensions=1024,
        use_mrl=False,
        batch_size=96,
    ),
    "fastembed_bge_768": ExperimentConfig(
        name="FastEmbed BGE-base 768d (local)",
        embedding_provider="fastembed",
        embedding_model="BAAI/bge-base-en-v1.5",
        dimensions=768,
        use_mrl=False,
        batch_size=32,
    ),
    "openai_256_quantized": ExperimentConfig(
        name="OpenAI 256d MRL + INT8 Quantization",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        dimensions=256,
        use_mrl=True,
        use_quantization=True,
        quantization_type="int8",
        batch_size=100,
    ),
    "openai_256_sparse": ExperimentConfig(
        name="OpenAI 256d MRL + BM25 Sparse",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        dimensions=256,
        use_mrl=True,
        use_sparse=True,
        sparse_model="bm25",
        batch_size=100,
    ),
}


class Embedder:
    """Unified embedder interface for different providers"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._client = None

    def embed(self, texts: List[str]) -> List[List[float]]:
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

        kwargs = {
            "model": self.config.embedding_model,
            "input": texts,
        }
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
    """Generate simple BM25 sparse vector"""
    tokens = text.lower().replace("[^\\w\\s]", " ").split()
    tokens = [t for t in tokens if len(t) > 2]

    term_freq: Dict[str, int] = {}
    for token in tokens:
        term_freq[token] = term_freq.get(token, 0) + 1

    indices = []
    values = []

    for token, freq in term_freq.items():
        # Simple hash function
        h = 0
        for c in token:
            h = ((h << 5) - h + ord(c)) & 0xFFFFFFFF
        index = h % 30000
        tf = freq / (freq + 1.2)
        indices.append(index)
        values.append(tf)

    return {"indices": indices, "values": values}


def ensure_experiments_collection(client: QdrantClient):
    """Ensure the experiments collection exists"""
    try:
        client.get_collection(EXPERIMENTS_COLLECTION)
    except:
        client.create_collection(
            collection_name=EXPERIMENTS_COLLECTION,
            vectors_config=VectorParams(size=1, distance=Distance.COSINE),  # Minimal vector
        )
        print(f"Created {EXPERIMENTS_COLLECTION} collection")


def store_experiment(client: QdrantClient, metrics: ExperimentMetrics):
    """Store experiment results in Qdrant"""
    ensure_experiments_collection(client)

    point = PointStruct(
        id=metrics.experiment_id,
        vector=[0.0],  # Placeholder - we use payload for data
        payload=metrics.to_dict(),
    )
    client.upsert(collection_name=EXPERIMENTS_COLLECTION, points=[point])


def list_experiments(client: QdrantClient) -> List[Dict]:
    """List all stored experiments"""
    ensure_experiments_collection(client)

    results = client.scroll(
        collection_name=EXPERIMENTS_COLLECTION,
        limit=100,
        with_payload=True,
    )

    experiments = []
    for point in results[0]:
        experiments.append(point.payload)

    return sorted(experiments, key=lambda x: x.get("started_at", ""), reverse=True)


def run_experiment(
    config: ExperimentConfig,
    sample_texts: List[str],
    collection_suffix: str = "",
) -> ExperimentMetrics:
    """Run a single embedding experiment"""

    experiment_id = uuid.uuid4().hex[:12]
    collection_name = f"experiment_{config.name.lower().replace(' ', '_')}_{experiment_id}"
    if collection_suffix:
        collection_name += f"_{collection_suffix}"

    metrics = ExperimentMetrics(
        experiment_id=experiment_id,
        config=asdict(config),
        started_at=datetime.utcnow().isoformat(),
        total_texts=len(sample_texts),
    )

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    embedder = Embedder(config)

    start_time = time.time()

    try:
        # Create collection with appropriate config
        vectors_config = {
            "dense": VectorParams(size=config.dimensions, distance=Distance.COSINE)
        }

        sparse_config = None
        if config.use_sparse:
            sparse_config = {"sparse": SparseVectorParams()}

        quantization = None
        if config.use_quantization:
            if config.quantization_type == "int8":
                quantization = ScalarQuantization(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )

        try:
            client.delete_collection(collection_name)
        except:
            pass

        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
            quantization_config=quantization,
        )

        print(f"\n[{config.name}] Created collection: {collection_name}")

        # Process in batches
        total_embedding_time = 0
        total_upsert_time = 0
        all_points = []

        batches = [
            sample_texts[i:i + config.batch_size]
            for i in range(0, len(sample_texts), config.batch_size)
        ]
        metrics.total_batches = len(batches)

        for batch_idx, batch in enumerate(batches):
            # Embed batch
            embed_start = time.time()
            vectors = embedder.embed(batch)
            embed_time = (time.time() - embed_start) * 1000
            total_embedding_time += embed_time

            # Create points
            points = []
            for i, (text, vector) in enumerate(zip(batch, vectors)):
                point_id = uuid.uuid4().hex
                point_vectors = {"dense": vector}

                if config.use_sparse:
                    sparse = generate_bm25_sparse(text)
                    point_vectors["sparse"] = {
                        "indices": sparse["indices"],
                        "values": sparse["values"],
                    }

                points.append(PointStruct(
                    id=point_id,
                    vector=point_vectors,
                    payload={"text": text[:500], "batch": batch_idx},
                ))

            # Upsert batch
            upsert_start = time.time()
            client.upsert(collection_name=collection_name, points=points)
            upsert_time = (time.time() - upsert_start) * 1000
            total_upsert_time += upsert_time

            all_points.extend(points)

            # Progress
            progress = (batch_idx + 1) / len(batches) * 100
            print(f"  [{config.name}] Batch {batch_idx + 1}/{len(batches)} "
                  f"({progress:.0f}%) - Embed: {embed_time:.0f}ms, Upsert: {upsert_time:.0f}ms")

        total_time = (time.time() - start_time) * 1000

        # Calculate metrics
        metrics.total_vectors = len(all_points)
        metrics.total_embedding_time_ms = total_embedding_time
        metrics.total_upsert_time_ms = total_upsert_time
        metrics.total_time_ms = total_time
        metrics.avg_embedding_time_per_batch_ms = total_embedding_time / len(batches)
        metrics.avg_upsert_time_per_batch_ms = total_upsert_time / len(batches)
        metrics.avg_embedding_time_per_vector_ms = total_embedding_time / len(all_points)
        metrics.vectors_per_second = len(all_points) / (total_time / 1000)

        # Estimate size
        bytes_per_float = 4
        if config.use_quantization and config.quantization_type == "int8":
            bytes_per_float = 1
        metrics.estimated_vector_size_bytes = config.dimensions * bytes_per_float
        metrics.estimated_total_size_mb = (
            metrics.estimated_vector_size_bytes * len(all_points)
        ) / (1024 * 1024)

        # Run search latency test
        if len(all_points) > 0:
            search_latencies = []
            test_queries = sample_texts[:10]
            test_vectors = embedder.embed(test_queries)

            for vec in test_vectors:
                search_start = time.time()
                client.query_points(
                    collection_name=collection_name,
                    query=vec,
                    using="dense",
                    limit=10,
                )
                search_latencies.append((time.time() - search_start) * 1000)

            search_latencies.sort()
            metrics.search_latency_p50_ms = search_latencies[len(search_latencies) // 2]
            metrics.search_latency_p95_ms = search_latencies[int(len(search_latencies) * 0.95)]

        metrics.status = "completed"
        metrics.completed_at = datetime.utcnow().isoformat()

        print(f"\n[{config.name}] COMPLETED:")
        print(f"  Vectors: {metrics.total_vectors}")
        print(f"  Total time: {metrics.total_time_ms:.0f}ms")
        print(f"  Throughput: {metrics.vectors_per_second:.1f} vectors/sec")
        print(f"  Avg embed time: {metrics.avg_embedding_time_per_vector_ms:.2f}ms/vector")
        print(f"  Estimated size: {metrics.estimated_total_size_mb:.2f}MB")
        print(f"  Search P50: {metrics.search_latency_p50_ms:.1f}ms")

    except Exception as e:
        metrics.status = "failed"
        metrics.errors.append(str(e))
        metrics.completed_at = datetime.utcnow().isoformat()
        print(f"\n[{config.name}] FAILED: {e}")

    # Store results
    store_experiment(client, metrics)

    return metrics


def compare_experiments(exp1: Dict, exp2: Dict):
    """Compare two experiments side by side"""
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON")
    print("=" * 70)

    fields = [
        ("Config", "config.name"),
        ("Provider", "config.embedding_provider"),
        ("Model", "config.embedding_model"),
        ("Dimensions", "config.dimensions"),
        ("MRL", "config.use_mrl"),
        ("Quantization", "config.use_quantization"),
        ("Sparse", "config.use_sparse"),
        ("Status", "status"),
        ("Total Vectors", "total_vectors"),
        ("Total Time (ms)", "total_time_ms"),
        ("Vectors/sec", "vectors_per_second"),
        ("Embed Time/vec (ms)", "avg_embedding_time_per_vector_ms"),
        ("Search P50 (ms)", "search_latency_p50_ms"),
        ("Search P95 (ms)", "search_latency_p95_ms"),
        ("Est. Size (MB)", "estimated_total_size_mb"),
    ]

    def get_nested(d: Dict, path: str):
        parts = path.split(".")
        val = d
        for p in parts:
            if isinstance(val, dict):
                val = val.get(p)
            else:
                return None
        return val

    print(f"\n{'Metric':<30} {'Experiment 1':<20} {'Experiment 2':<20} {'Diff':<15}")
    print("-" * 85)

    for label, path in fields:
        v1 = get_nested(exp1, path)
        v2 = get_nested(exp2, path)

        diff = ""
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v1 != 0:
            pct = ((v2 - v1) / v1) * 100
            diff = f"{pct:+.1f}%"

        v1_str = f"{v1:.2f}" if isinstance(v1, float) else str(v1)
        v2_str = f"{v2:.2f}" if isinstance(v2, float) else str(v2)

        print(f"{label:<30} {v1_str:<20} {v2_str:<20} {diff:<15}")


def main():
    parser = argparse.ArgumentParser(description="Run embedding experiments")
    parser.add_argument("--configs", nargs="+", help="Config names to run", choices=list(CONFIGS.keys()))
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--compare", nargs=2, help="Compare two experiment IDs")
    parser.add_argument("--sample-size", type=int, default=500, help="Number of sample texts")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel experiments")

    args = parser.parse_args()

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

    if args.list:
        experiments = list_experiments(client)
        print("\n" + "=" * 80)
        print("STORED EXPERIMENTS")
        print("=" * 80)

        for exp in experiments:
            config = exp.get("config", {})
            print(f"\n  ID: {exp['experiment_id']}")
            print(f"  Config: {config.get('name', 'unknown')}")
            print(f"  Status: {exp['status']}")
            print(f"  Started: {exp['started_at']}")
            print(f"  Vectors: {exp.get('total_vectors', 0)}")
            print(f"  Throughput: {exp.get('vectors_per_second', 0):.1f} vec/s")
            print(f"  Search P50: {exp.get('search_latency_p50_ms', 'N/A')}ms")
        return

    if args.compare:
        experiments = list_experiments(client)
        exp_map = {e["experiment_id"]: e for e in experiments}

        if args.compare[0] not in exp_map or args.compare[1] not in exp_map:
            print("Error: Experiment ID not found")
            print("Available IDs:", list(exp_map.keys()))
            return

        compare_experiments(exp_map[args.compare[0]], exp_map[args.compare[1]])
        return

    if not args.configs:
        print("Available configs:")
        for name, config in CONFIGS.items():
            print(f"  {name}: {config.name}")
        print("\nUsage: python ingestion_experiments.py --configs openai_256_mrl cohere_256_mrl")
        return

    # Generate sample texts
    print(f"\nGenerating {args.sample_size} sample texts...")
    sample_texts = [
        f"Sample document {i}: This is a test document about various topics "
        f"including technology, science, and culture. Document number {i}."
        for i in range(args.sample_size)
    ]

    # Run experiments
    configs_to_run = [CONFIGS[name] for name in args.configs]

    if args.parallel > 1 and len(configs_to_run) > 1:
        print(f"\nRunning {len(configs_to_run)} experiments in parallel (max {args.parallel})...")
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_experiment, config, sample_texts): config
                for config in configs_to_run
            }
            results = []
            for future in as_completed(futures):
                results.append(future.result())
    else:
        print(f"\nRunning {len(configs_to_run)} experiments sequentially...")
        results = [run_experiment(config, sample_texts) for config in configs_to_run]

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    for metrics in results:
        config_name = metrics.config.get("name", "unknown")
        print(f"\n{config_name}:")
        print(f"  ID: {metrics.experiment_id}")
        print(f"  Status: {metrics.status}")
        print(f"  Vectors: {metrics.total_vectors}")
        print(f"  Throughput: {metrics.vectors_per_second:.1f} vectors/sec")
        print(f"  Search P50: {metrics.search_latency_p50_ms:.1f}ms" if metrics.search_latency_p50_ms else "  Search P50: N/A")
        print(f"  Est. Size: {metrics.estimated_total_size_mb:.2f}MB")


if __name__ == "__main__":
    main()
