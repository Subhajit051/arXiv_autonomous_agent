import random
import time

from google import genai
from google.genai import types

from .config import GEMINI_API_KEY, GEMINI_MODEL, EMBEDDING_MODEL


if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Create a .env file from .env.example "
        "and add your Gemini API key."
    )


client = genai.Client(api_key=GEMINI_API_KEY)


# The configured model is tried first, followed by stable fallback models.
TEXT_MODELS = []
for model in [GEMINI_MODEL, "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]:
    if model and model not in TEXT_MODELS:
        TEXT_MODELS.append(model)


def _is_retryable_error(error: Exception) -> bool:
    """Return True for transient Gemini service/rate-limit errors."""
    message = str(error).upper()
    retryable_codes = ("408", "429", "500", "502", "503", "504")
    retryable_messages = (
        "TIMEOUT",
        "DEADLINE EXCEEDED",
        "TOO MANY REQUESTS",
        "RESOURCE EXHAUSTED",
        "INTERNAL SERVER ERROR",
        "BAD GATEWAY",
        "SERVICE UNAVAILABLE",
        "GATEWAY TIMEOUT",
        "UNAVAILABLE",
    )
    return any(code in message for code in retryable_codes) or any(
        text in message for text in retryable_messages
    )


def _generate_with_model(prompt: str, model: str) -> str:
    """Generate with limited application-level retries.

    The Google GenAI SDK already retries transient failures internally, so
    the application adds only a small second layer before falling back.
    """
    max_attempts = 2

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            if not response.text:
                raise RuntimeError(f"Gemini model '{model}' returned an empty response.")
            return response.text.strip()
        except Exception as error:
            if not _is_retryable_error(error):
                raise
            if attempt == max_attempts:
                raise

            delay = (2 ** attempt) + random.uniform(0.2, 1.0)
            print(
                f"\nGemini model '{model}' temporarily unavailable "
                f"(attempt {attempt}/{max_attempts}). "
                f"Retrying in {delay:.1f} seconds..."
            )
            time.sleep(delay)

    raise RuntimeError(f"Gemini model '{model}' failed.")


def generate(prompt: str) -> str:
    """Generate text with controlled retry and model fallback."""
    last_error = None

    for index, model in enumerate(TEXT_MODELS):
        try:
            if index > 0:
                print(f"\nTrying fallback Gemini model: {model}")
            return _generate_with_model(prompt, model)
        except Exception as error:
            last_error = error
            if not _is_retryable_error(error):
                raise
            print(f"\nGemini model '{model}' was unavailable after its retry attempts.")

    raise RuntimeError(
        "All configured Gemini text-generation models are currently unavailable. "
        "Please try the request again later."
    ) from last_error


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed each document separately so every chunk gets one vector."""
    embeddings = []
    for text in texts:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            ),
        )
        embeddings.append(response.embeddings[0].values)
    return embeddings


def embed_query(text: str) -> list[float]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )
    return response.embeddings[0].values
