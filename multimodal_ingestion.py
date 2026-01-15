"""
Multi-Modal Ingestion for Qdrant

Supports:
- Text embeddings (OpenAI, Cohere, FastEmbed)
- Image embeddings (CLIP, OpenCLIP)
- ColBERT/ColPali late interaction (multi-vector per document)
- Named vectors (text + image in same collection)

Based on:
- https://qdrant.tech/documentation/concepts/vectors/
- https://qdrant.tech/documentation/fastembed/fastembed-colbert/
- https://qdrant.tech/blog/qdrant-colpali/

Usage:
    # Text + Image (CLIP)
    python multimodal_ingestion.py --parquet wiki.parquet --config clip_multimodal

    # ColBERT late interaction
    python multimodal_ingestion.py --parquet wiki.parquet --config colbert_late

    # ColPali for PDFs/images
    python multimodal_ingestion.py --images ./pages --config colpali

    # Compare configs
    python multimodal_ingestion.py --results
"""

import os
import sys
import time
import uuid
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    MultiVectorConfig,
    MultiVectorComparator,
    HnswConfigDiff,
    ScalarQuantization,
    ScalarType,
    BinaryQuantization,
    BinaryQuantizationConfig,
)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EXPERIMENTS_COLLECTION = "ingestion_experiments"


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal ingestion"""
    name: str

    # Text embedding
    text_provider: str  # openai, cohere, fastembed, none
    text_model: str
    text_dimensions: int
    text_use_mrl: bool = False

    # Image embedding
    image_provider: str  # clip, openclip, none
    image_model: str = ""
    image_dimensions: int = 0

    # Late interaction (ColBERT/ColPali)
    use_late_interaction: bool = False
    late_interaction_model: str = ""  # colbert, colpali, colqwen
    late_interaction_dimensions: int = 0

    # Quantization
    use_quantization: bool = False
    quantization_type: str = ""  # int8, binary

    # Processing
    batch_size: int = 100
    text_column: str = "text"
    image_column: str = "image_url"  # or "image_path"

    # Qdrant settings
    defer_indexing: bool = True
    on_disk: bool = True


@dataclass
class MultiModalMetrics:
    """Metrics for multi-modal ingestion"""
    experiment_id: str
    config_name: str
    config: Dict[str, Any]
    source_path: str
    collection_name: str
    started_at: str
    completed_at: Optional[str] = None
    status: str = "running"

    total_items: int = 0
    total_text_vectors: int = 0
    total_image_vectors: int = 0
    total_multivectors: int = 0

    text_embedding_time_ms: float = 0
    image_embedding_time_ms: float = 0
    late_interaction_time_ms: float = 0
    upsert_time_ms: float = 0
    total_time_ms: float = 0

    items_per_second: float = 0
    search_latency_p50_ms: Optional[float] = None
    search_latency_p95_ms: Optional[float] = None

    estimated_size_mb: float = 0
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Predefined Configurations
# =============================================================================

CONFIGS: Dict[str, MultiModalConfig] = {
    # Text-only configs
    "text_openai_256": MultiModalConfig(
        name="Text Only - OpenAI 256d MRL",
        text_provider="openai",
        text_model="text-embedding-3-small",
        text_dimensions=256,
        text_use_mrl=True,
        image_provider="none",
        batch_size=500,
    ),

    # CLIP multi-modal (same embedding space for text + image)
    "clip_multimodal": MultiModalConfig(
        name="CLIP Multi-Modal (Text + Image)",
        text_provider="clip",
        text_model="ViT-B/32",
        text_dimensions=512,
        image_provider="clip",
        image_model="ViT-B/32",
        image_dimensions=512,
        batch_size=100,
    ),

    "openclip_large": MultiModalConfig(
        name="OpenCLIP Large (Text + Image)",
        text_provider="openclip",
        text_model="ViT-L-14",
        text_dimensions=768,
        image_provider="openclip",
        image_model="ViT-L-14",
        image_dimensions=768,
        batch_size=50,
    ),

    # Named vectors: separate text + image vectors
    "named_text_image": MultiModalConfig(
        name="Named Vectors (OpenAI Text + CLIP Image)",
        text_provider="openai",
        text_model="text-embedding-3-small",
        text_dimensions=256,
        text_use_mrl=True,
        image_provider="clip",
        image_model="ViT-B/32",
        image_dimensions=512,
        batch_size=100,
    ),

    # ColBERT late interaction
    "colbert_late": MultiModalConfig(
        name="ColBERT Late Interaction",
        text_provider="none",
        text_model="",
        text_dimensions=0,
        image_provider="none",
        use_late_interaction=True,
        late_interaction_model="colbert",
        late_interaction_dimensions=128,  # ColBERT uses 128d per token
        batch_size=50,
    ),

    "colbert_quantized": MultiModalConfig(
        name="ColBERT + Binary Quantization",
        text_provider="none",
        text_model="",
        text_dimensions=0,
        image_provider="none",
        use_late_interaction=True,
        late_interaction_model="colbert",
        late_interaction_dimensions=128,
        use_quantization=True,
        quantization_type="binary",
        batch_size=50,
    ),

    # ColPali for documents/PDFs
    "colpali": MultiModalConfig(
        name="ColPali (Document Images)",
        text_provider="none",
        text_model="",
        text_dimensions=0,
        image_provider="none",
        use_late_interaction=True,
        late_interaction_model="colpali",
        late_interaction_dimensions=128,
        batch_size=10,  # ColPali generates 1000+ vectors per image
    ),

    "colqwen": MultiModalConfig(
        name="ColQwen (Document Images)",
        text_provider="none",
        text_model="",
        text_dimensions=0,
        image_provider="none",
        use_late_interaction=True,
        late_interaction_model="colqwen",
        late_interaction_dimensions=128,
        batch_size=10,
    ),

    # Hybrid: Dense + Late Interaction
    "hybrid_dense_colbert": MultiModalConfig(
        name="Hybrid: OpenAI Dense + ColBERT",
        text_provider="openai",
        text_model="text-embedding-3-small",
        text_dimensions=256,
        text_use_mrl=True,
        image_provider="none",
        use_late_interaction=True,
        late_interaction_model="colbert",
        late_interaction_dimensions=128,
        batch_size=50,
    ),

    # Full multi-modal: Text + Image + Late Interaction
    "full_multimodal": MultiModalConfig(
        name="Full Multi-Modal (Text + Image + ColBERT)",
        text_provider="openai",
        text_model="text-embedding-3-small",
        text_dimensions=256,
        text_use_mrl=True,
        image_provider="clip",
        image_model="ViT-B/32",
        image_dimensions=512,
        use_late_interaction=True,
        late_interaction_model="colbert",
        late_interaction_dimensions=128,
        batch_size=25,
    ),
}


# =============================================================================
# Embedding Functions
# =============================================================================

class TextEmbedder:
    """Text embedding with various providers"""

    def __init__(self, provider: str, model: str, dimensions: int, use_mrl: bool):
        self.provider = provider
        self.model = model
        self.dimensions = dimensions
        self.use_mrl = use_mrl
        self._client = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "openai":
            return self._embed_openai(texts)
        elif self.provider == "cohere":
            return self._embed_cohere(texts)
        elif self.provider == "clip":
            return self._embed_clip_text(texts)
        elif self.provider == "openclip":
            return self._embed_openclip_text(texts)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        import openai
        if not self._client:
            self._client = openai.OpenAI(api_key=OPENAI_API_KEY)

        kwargs = {"model": self.model, "input": texts}
        if self.use_mrl:
            kwargs["dimensions"] = self.dimensions

        response = self._client.embeddings.create(**kwargs)
        return [d.embedding for d in response.data]

    def _embed_cohere(self, texts: List[str]) -> List[List[float]]:
        import cohere
        if not self._client:
            self._client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

        kwargs = {"model": self.model, "texts": texts, "input_type": "search_document"}
        if self.use_mrl:
            kwargs["output_dimension"] = self.dimensions

        response = self._client.embed(**kwargs)
        return [list(e) for e in response.embeddings]

    def _embed_clip_text(self, texts: List[str]) -> List[List[float]]:
        try:
            import clip
            import torch
        except ImportError:
            raise ImportError("pip install git+https://github.com/openai/CLIP.git")

        if not self._client:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._client, self._preprocess = clip.load(self.model, device=device)
            self._device = device

        with torch.no_grad():
            tokens = clip.tokenize(texts, truncate=True).to(self._device)
            embeddings = self._client.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            return embeddings.cpu().numpy().tolist()

    def _embed_openclip_text(self, texts: List[str]) -> List[List[float]]:
        try:
            import open_clip
            import torch
        except ImportError:
            raise ImportError("pip install open_clip_torch")

        if not self._client:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._client, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model, pretrained="openai"
            )
            self._tokenizer = open_clip.get_tokenizer(self.model)
            self._device = device
            self._client = self._client.to(device)

        with torch.no_grad():
            tokens = self._tokenizer(texts).to(self._device)
            embeddings = self._client.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            return embeddings.cpu().numpy().tolist()


class ImageEmbedder:
    """Image embedding with CLIP/OpenCLIP"""

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self._client = None

    def embed(self, images: List[Any]) -> List[List[float]]:
        """Embed images (PIL Images or paths)"""
        if self.provider == "clip":
            return self._embed_clip(images)
        elif self.provider == "openclip":
            return self._embed_openclip(images)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _embed_clip(self, images: List[Any]) -> List[List[float]]:
        try:
            import clip
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError("pip install git+https://github.com/openai/CLIP.git pillow")

        if not self._client:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._client, self._preprocess = clip.load(self.model, device=device)
            self._device = device

        processed = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            processed.append(self._preprocess(img))

        with torch.no_grad():
            batch = torch.stack(processed).to(self._device)
            embeddings = self._client.encode_image(batch)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            return embeddings.cpu().numpy().tolist()

    def _embed_openclip(self, images: List[Any]) -> List[List[float]]:
        try:
            import open_clip
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError("pip install open_clip_torch pillow")

        if not self._client:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._client, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model, pretrained="openai"
            )
            self._device = device
            self._client = self._client.to(device)

        processed = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            processed.append(self._preprocess(img))

        with torch.no_grad():
            batch = torch.stack(processed).to(self._device)
            embeddings = self._client.encode_image(batch)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            return embeddings.cpu().numpy().tolist()


class LateInteractionEmbedder:
    """ColBERT/ColPali multi-vector embeddings"""

    def __init__(self, model_type: str):
        self.model_type = model_type
        self._model = None

    def embed_texts(self, texts: List[str]) -> List[List[List[float]]]:
        """Returns list of [num_tokens, dim] per text"""
        if self.model_type == "colbert":
            return self._embed_colbert(texts)
        else:
            raise ValueError(f"Text embedding not supported for {self.model_type}")

    def embed_images(self, images: List[Any]) -> List[List[List[float]]]:
        """Returns list of [num_patches, dim] per image"""
        if self.model_type in ["colpali", "colqwen"]:
            return self._embed_colpali(images)
        else:
            raise ValueError(f"Image embedding not supported for {self.model_type}")

    def _embed_colbert(self, texts: List[str]) -> List[List[List[float]]]:
        """Embed with ColBERT - returns multi-vectors per document"""
        try:
            from fastembed import LateInteractionTextEmbedding
        except ImportError:
            raise ImportError("pip install fastembed")

        if not self._model:
            self._model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

        # Returns list of 2D arrays [num_tokens, 128]
        embeddings = list(self._model.embed(texts))
        return [e.tolist() for e in embeddings]

    def _embed_colpali(self, images: List[Any]) -> List[List[List[float]]]:
        """Embed document images with ColPali"""
        try:
            from colpali_engine.models import ColPali
            from colpali_engine.utils.torch_utils import get_torch_device
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError("pip install colpali-engine")

        if not self._model:
            device = get_torch_device("auto")
            model_name = "vidore/colpali-v1.2" if self.model_type == "colpali" else "vidore/colqwen2-v0.1"
            self._model = ColPali.from_pretrained(model_name).to(device).eval()
            self._device = device

        processed = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            processed.append(img)

        with torch.no_grad():
            # ColPali returns [batch, num_patches, dim]
            embeddings = self._model.forward(processed)
            return [e.cpu().numpy().tolist() for e in embeddings]


# =============================================================================
# Collection Creation
# =============================================================================

def create_multimodal_collection(
    client: QdrantClient,
    collection_name: str,
    config: MultiModalConfig,
) -> None:
    """Create collection with appropriate vector configuration"""

    vectors_config = {}

    # Dense text vector
    if config.text_provider not in ["none", ""]:
        vectors_config["text"] = VectorParams(
            size=config.text_dimensions,
            distance=Distance.COSINE,
            on_disk=config.on_disk,
        )

    # Dense image vector
    if config.image_provider not in ["none", ""]:
        vectors_config["image"] = VectorParams(
            size=config.image_dimensions,
            distance=Distance.COSINE,
            on_disk=config.on_disk,
        )

    # Late interaction multi-vector
    if config.use_late_interaction:
        multivector_config = MultiVectorConfig(
            comparator=MultiVectorComparator.MAX_SIM,
        )
        vectors_config["late"] = VectorParams(
            size=config.late_interaction_dimensions,
            distance=Distance.COSINE,
            on_disk=config.on_disk,
            multivector_config=multivector_config,
        )

    # Quantization
    quantization = None
    if config.use_quantization:
        if config.quantization_type == "int8":
            quantization = ScalarQuantization(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )
        elif config.quantization_type == "binary":
            quantization = BinaryQuantization(
                binary=BinaryQuantizationConfig(always_ram=True),
            )

    # HNSW config (disable for multi-vectors as per Qdrant docs)
    hnsw_config = None
    if config.defer_indexing or config.use_late_interaction:
        hnsw_config = HnswConfigDiff(m=0)

    # Delete existing
    try:
        client.delete_collection(collection_name)
    except:
        pass

    # Create
    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        quantization_config=quantization,
        hnsw_config=hnsw_config,
    )

    print(f"Created collection: {collection_name}")
    print(f"  Vectors: {list(vectors_config.keys())}")
    print(f"  Quantization: {config.quantization_type or 'none'}")


# =============================================================================
# Ingestion
# =============================================================================

def run_multimodal_ingestion(
    config: MultiModalConfig,
    parquet_path: Optional[Path] = None,
    image_dir: Optional[Path] = None,
) -> MultiModalMetrics:
    """Run multi-modal ingestion"""

    experiment_id = uuid.uuid4().hex[:12]
    collection_name = f"mm_{config.name.lower().replace(' ', '_')[:25]}_{experiment_id}"

    source = str(parquet_path or image_dir or "unknown")
    metrics = MultiModalMetrics(
        experiment_id=experiment_id,
        config_name=config.name,
        config=asdict(config),
        source_path=source,
        collection_name=collection_name,
        started_at=datetime.utcnow().isoformat(),
    )

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=300)

    # Initialize embedders
    text_embedder = None
    image_embedder = None
    late_embedder = None

    if config.text_provider not in ["none", ""]:
        text_embedder = TextEmbedder(
            config.text_provider,
            config.text_model,
            config.text_dimensions,
            config.text_use_mrl,
        )

    if config.image_provider not in ["none", ""]:
        image_embedder = ImageEmbedder(config.image_provider, config.image_model)

    if config.use_late_interaction:
        late_embedder = LateInteractionEmbedder(config.late_interaction_model)

    total_start = time.time()

    try:
        # Create collection
        create_multimodal_collection(client, collection_name, config)

        # Load data
        if parquet_path:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(parquet_path)
            metrics.total_items = parquet_file.metadata.num_rows

            # Process batches
            batch_count = 0
            for batch in parquet_file.iter_batches(batch_size=config.batch_size):
                batch_count += 1
                texts = batch[config.text_column].to_pylist() if config.text_column in batch.column_names else []
                texts = [str(t) for t in texts if t]

                images = []
                if config.image_column in batch.column_names:
                    images = [p for p in batch[config.image_column].to_pylist() if p]

                # Embed and upsert
                points = process_batch(
                    texts, images, text_embedder, image_embedder, late_embedder,
                    config, metrics, batch_count
                )

                if points:
                    upsert_start = time.time()
                    client.upsert(collection_name=collection_name, points=points, wait=True)
                    metrics.upsert_time_ms += (time.time() - upsert_start) * 1000

                if batch_count % 10 == 0:
                    print(f"  Batch {batch_count}: {metrics.total_text_vectors} text, "
                          f"{metrics.total_image_vectors} image, {metrics.total_multivectors} multi")

        elif image_dir:
            # Process images from directory
            image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
            metrics.total_items = len(image_files)

            for i in range(0, len(image_files), config.batch_size):
                batch_images = image_files[i:i + config.batch_size]
                points = process_batch(
                    [], batch_images, text_embedder, image_embedder, late_embedder,
                    config, metrics, i // config.batch_size
                )

                if points:
                    upsert_start = time.time()
                    client.upsert(collection_name=collection_name, points=points, wait=True)
                    metrics.upsert_time_ms += (time.time() - upsert_start) * 1000

        # Re-enable indexing for dense vectors
        if config.defer_indexing and not config.use_late_interaction:
            print("  Re-enabling HNSW indexing...")
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
        metrics.total_time_ms = total_time
        metrics.items_per_second = metrics.total_items / (total_time / 1000) if total_time > 0 else 0

        # Estimate size
        text_size = config.text_dimensions * 4 * metrics.total_text_vectors if config.text_dimensions else 0
        image_size = config.image_dimensions * 4 * metrics.total_image_vectors if config.image_dimensions else 0
        late_size = config.late_interaction_dimensions * 4 * metrics.total_multivectors * 100  # ~100 tokens avg
        metrics.estimated_size_mb = (text_size + image_size + late_size) / (1024 * 1024)

        metrics.status = "completed"
        metrics.completed_at = datetime.utcnow().isoformat()

        print(f"\n[{config.name}] COMPLETED:")
        print(f"  Items: {metrics.total_items}")
        print(f"  Text vectors: {metrics.total_text_vectors}")
        print(f"  Image vectors: {metrics.total_image_vectors}")
        print(f"  Multi-vectors: {metrics.total_multivectors}")
        print(f"  Throughput: {metrics.items_per_second:.1f} items/sec")

    except Exception as e:
        metrics.status = "failed"
        metrics.errors.append(str(e))
        metrics.completed_at = datetime.utcnow().isoformat()
        print(f"[{config.name}] FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Store results
    store_metrics(client, metrics)

    return metrics


def process_batch(
    texts: List[str],
    images: List[Any],
    text_embedder: Optional[TextEmbedder],
    image_embedder: Optional[ImageEmbedder],
    late_embedder: Optional[LateInteractionEmbedder],
    config: MultiModalConfig,
    metrics: MultiModalMetrics,
    batch_idx: int,
) -> List[PointStruct]:
    """Process a batch of texts/images into points"""

    points = []

    # Get embeddings
    text_vectors = []
    if text_embedder and texts:
        start = time.time()
        text_vectors = text_embedder.embed(texts)
        metrics.text_embedding_time_ms += (time.time() - start) * 1000

    image_vectors = []
    if image_embedder and images:
        start = time.time()
        image_vectors = image_embedder.embed(images)
        metrics.image_embedding_time_ms += (time.time() - start) * 1000

    late_vectors = []
    if late_embedder:
        start = time.time()
        if texts and config.late_interaction_model == "colbert":
            late_vectors = late_embedder.embed_texts(texts)
        elif images and config.late_interaction_model in ["colpali", "colqwen"]:
            late_vectors = late_embedder.embed_images(images)
        metrics.late_interaction_time_ms += (time.time() - start) * 1000

    # Create points
    num_items = max(len(texts), len(images))
    for i in range(num_items):
        point_id = uuid.uuid4().hex
        vectors = {}
        payload = {"batch": batch_idx}

        if i < len(text_vectors):
            vectors["text"] = text_vectors[i]
            payload["text"] = texts[i][:1000] if i < len(texts) else ""
            metrics.total_text_vectors += 1

        if i < len(image_vectors):
            vectors["image"] = image_vectors[i]
            payload["image_path"] = str(images[i]) if i < len(images) else ""
            metrics.total_image_vectors += 1

        if i < len(late_vectors):
            vectors["late"] = late_vectors[i]  # List of token vectors
            metrics.total_multivectors += 1

        if vectors:
            points.append(PointStruct(
                id=point_id,
                vector=vectors,
                payload=payload,
            ))

    return points


def store_metrics(client: QdrantClient, metrics: MultiModalMetrics):
    """Store experiment metrics"""
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
        payload=asdict(metrics),
    )
    client.upsert(collection_name=EXPERIMENTS_COLLECTION, points=[point])


def list_results() -> List[Dict]:
    """List all experiment results"""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    try:
        results = client.scroll(
            collection_name=EXPERIMENTS_COLLECTION,
            limit=100,
            with_payload=True,
        )
        return sorted(
            [p.payload for p in results[0]],
            key=lambda x: x.get("started_at", ""),
            reverse=True,
        )
    except:
        return []


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-modal Qdrant ingestion")
    parser.add_argument("--parquet", type=Path, help="Parquet file to ingest")
    parser.add_argument("--images", type=Path, help="Image directory to ingest")
    parser.add_argument("--config", choices=list(CONFIGS.keys()), help="Config to use")
    parser.add_argument("--list-configs", action="store_true", help="List configs")
    parser.add_argument("--results", action="store_true", help="Show results")

    args = parser.parse_args()

    if args.list_configs:
        print("\nAvailable Multi-Modal Configurations:")
        print("=" * 70)
        for name, cfg in CONFIGS.items():
            print(f"\n{name}:")
            print(f"  {cfg.name}")
            print(f"  Text: {cfg.text_provider} {cfg.text_model} ({cfg.text_dimensions}d)")
            print(f"  Image: {cfg.image_provider} {cfg.image_model} ({cfg.image_dimensions}d)")
            if cfg.use_late_interaction:
                print(f"  Late Interaction: {cfg.late_interaction_model} ({cfg.late_interaction_dimensions}d)")
        return

    if args.results:
        results = list_results()
        print("\n" + "=" * 100)
        print("MULTI-MODAL EXPERIMENT RESULTS")
        print("=" * 100)

        headers = ["Config", "Text", "Image", "Multi", "Items/s", "Size MB", "Status"]
        print("  ".join(f"{h:<18}" for h in headers))
        print("-" * 130)

        for r in results:
            row = [
                r.get("config_name", "?")[:16],
                str(r.get("total_text_vectors", 0)),
                str(r.get("total_image_vectors", 0)),
                str(r.get("total_multivectors", 0)),
                f"{r.get('items_per_second', 0):.1f}",
                f"{r.get('estimated_size_mb', 0):.2f}",
                r.get("status", "?"),
            ]
            print("  ".join(f"{v:<18}" for v in row))
        return

    if not args.config:
        print("Error: Must specify --config")
        print("Use --list-configs to see available options")
        return

    if not args.parquet and not args.images:
        print("Error: Must specify --parquet or --images")
        return

    config = CONFIGS[args.config]
    run_multimodal_ingestion(config, args.parquet, args.images)


if __name__ == "__main__":
    main()
