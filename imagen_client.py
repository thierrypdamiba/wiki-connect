"""
Google AI Image Generation Client

Supports multiple image generation models:
- Nano Banana Pro (Gemini 3 Pro Image) - Best quality, multimodal
- Imagen 4.0 Fast - Fast generation
- Imagen 4.0 Ultra - Highest quality
"""

import os
import base64
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

from google import genai
from google.genai import types

# Initialize client
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

# Available image models
IMAGE_MODELS = {
    "nano-banana": "models/nano-banana-pro-preview",  # Gemini 3 Pro Image - best quality
    "imagen-fast": "imagen-4.0-fast-generate-001",    # Fast Imagen 4
    "imagen-ultra": "imagen-4.0-ultra-generate-001",  # Ultra quality Imagen 4
}

# Default model
DEFAULT_IMAGE_MODEL = "nano-banana"


async def generate_image_nano_banana(
    prompt: str,
) -> tuple[bytes | None, str]:
    """
    Generate an image using Nano Banana Pro (Gemini 3 Pro Image).

    This is the best quality model with multimodal understanding.

    Args:
        prompt: Text description of the image to generate

    Returns:
        Tuple of (image_bytes, mime_type) or (None, "") if generation fails
    """
    try:
        response = await client.aio.models.generate_content(
            model=IMAGE_MODELS["nano-banana"],
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            )
        )

        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    return part.inline_data.data, part.inline_data.mime_type

        return None, ""

    except Exception as e:
        print(f"Nano Banana error: {e}")
        return None, ""


def generate_image_nano_banana_sync(prompt: str) -> tuple[bytes | None, str]:
    """Synchronous version of Nano Banana image generation."""
    try:
        response = client.models.generate_content(
            model=IMAGE_MODELS["nano-banana"],
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            )
        )

        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    return part.inline_data.data, part.inline_data.mime_type

        return None, ""

    except Exception as e:
        print(f"Nano Banana error: {e}")
        return None, ""


async def generate_image(
    prompt: str,
    model: Literal["nano-banana", "imagen-fast", "imagen-ultra"] = DEFAULT_IMAGE_MODEL,
    aspect_ratio: str = "1:1",
    negative_prompt: str = None,
) -> bytes | None:
    """
    Generate an image using specified model.

    Args:
        prompt: Text description of the image to generate
        model: Which model to use (nano-banana, imagen-fast, imagen-ultra)
        aspect_ratio: Image aspect ratio (for Imagen models only)
        negative_prompt: Things to avoid in the image (for Imagen models only)

    Returns:
        Image bytes or None if generation fails
    """
    if model == "nano-banana":
        image_bytes, _ = await generate_image_nano_banana(prompt)
        return image_bytes

    # Use Imagen for other models
    try:
        config = types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio=aspect_ratio,
            safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
            person_generation="DONT_ALLOW",
        )

        if negative_prompt:
            config.negative_prompt = negative_prompt

        response = await client.aio.models.generate_images(
            model=IMAGE_MODELS[model],
            prompt=prompt,
            config=config,
        )

        if response.generated_images:
            return response.generated_images[0].image.image_bytes

        return None

    except Exception as e:
        print(f"Imagen error: {e}")
        return None


def generate_image_sync(
    prompt: str,
    model: Literal["nano-banana", "imagen-fast", "imagen-ultra"] = DEFAULT_IMAGE_MODEL,
    aspect_ratio: str = "1:1",
    negative_prompt: str = None,
) -> bytes | None:
    """Synchronous version of generate_image."""
    if model == "nano-banana":
        image_bytes, _ = generate_image_nano_banana_sync(prompt)
        return image_bytes

    try:
        config = types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio=aspect_ratio,
            safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
            person_generation="DONT_ALLOW",
        )

        if negative_prompt:
            config.negative_prompt = negative_prompt

        response = client.models.generate_images(
            model=IMAGE_MODELS[model],
            prompt=prompt,
            config=config,
        )

        if response.generated_images:
            return response.generated_images[0].image.image_bytes

        return None

    except Exception as e:
        print(f"Imagen error: {e}")
        return None


def image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string for display."""
    return base64.b64encode(image_bytes).decode("utf-8")


def save_image(image_bytes: bytes, filepath: str) -> str:
    """Save image bytes to a file."""
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return filepath


# =============================================================================
# PROMPT TEMPLATES FOR WIKI CONNECT
# =============================================================================

def build_connection_image_prompt(
    topic_a: str,
    topic_b: str,
    connection_summary: str,
) -> str:
    """
    Build a prompt for generating an image that represents the connection
    between two Wikipedia topics.

    Args:
        topic_a: First topic (e.g., "Black Holes")
        topic_b: Second topic (e.g., "Quantum Computing")
        connection_summary: Brief description of how they connect

    Returns:
        Optimized prompt for Imagen
    """
    prompt = f"""Create an artistic, educational illustration showing the connection between "{topic_a}" and "{topic_b}".

The image should visually represent: {connection_summary}

Style: Clean, modern infographic style with vibrant colors. Scientific accuracy meets artistic beauty. No text or labels in the image. Abstract representation of concepts."""

    return prompt


def build_topic_illustration_prompt(topic: str, context: str = "") -> str:
    """Build a prompt for illustrating a single Wikipedia topic."""
    prompt = f"""Create an educational illustration of "{topic}".

{f'Context: {context}' if context else ''}

Style: Clean, modern, scientifically accurate. Vibrant colors on a clean background. No text or labels. Professional quality suitable for an encyclopedia."""

    return prompt
