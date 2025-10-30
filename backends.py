import urllib
import time
from typing import Dict, List, Tuple, Optional
import json

from usage_tracker import estimate_tokens


# ============================================================================
# Backend Abstraction Layer - Support Multiple Local Model Backends
# ============================================================================

class ModelBackend:
    """Abstract base for different local model backends"""

    async def generate(self, messages: List[Dict[str, str]], model: str) -> Tuple[str, int, int, float]:
        """
        Generate response from model.

        Returns:
            Tuple: (response_text, tokens_sent, tokens_received, duration_seconds)
        """
        raise NotImplementedError


class OllamaBackend(ModelBackend):
    """Backend for Ollama"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    async def generate(self, messages: List[Dict[str, str]], model: str) -> Tuple[str, int, int, float]:
        start_time = time.time()

        # Calculate input tokens
        input_text = " ".join(msg["content"] for msg in messages)
        tokens_sent = estimate_tokens(input_text)

        try:
            import ollama
            response = ollama.chat(model=model, messages=messages)
            output_text = response['message']['content']

            # Calculate output tokens
            tokens_received = estimate_tokens(output_text)
            duration = time.time() - start_time

            return output_text, tokens_sent, tokens_received, duration

        except ImportError:
            raise RuntimeError("Ollama package not installed. Run: pip install ollama")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {str(e)}")


class LMStudioBackend(ModelBackend):
    """Backend for LM Studio OpenAI-compatible API"""

    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        self.base_url = base_url

    async def generate(self, messages: List[Dict[str, str]], model: str) -> Tuple[str, int, int, float]:
        """Call LM Studio's OpenAI-compatible API"""
        start_time = time.time()

        # Calculate input tokens
        input_text = " ".join(msg["content"] for msg in messages)
        tokens_sent = estimate_tokens(input_text)

        request_data = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": -1,  # LM Studio uses -1 for unlimited
        }

        try:
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(request_data).encode('utf-8'),
                headers={
                    "Content-Type": "application/json",
                }
            )

            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))
                output_text = result['choices'][0]['message']['content']

                # Calculate output tokens
                tokens_received = estimate_tokens(output_text)
                duration = time.time() - start_time

                return output_text, tokens_sent, tokens_received, duration

        except urllib.error.URLError as e:
            raise RuntimeError(f"LM Studio connection error: {str(e)}. Is LM Studio running on {self.base_url}?")
        except Exception as e:
            raise RuntimeError(f"LM Studio error: {str(e)}")


class OpenAICompatibleBackend(ModelBackend):
    """Backend for any OpenAI-compatible API (vLLM, LocalAI, etc.)"""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key

    async def generate(self, messages: List[Dict[str, str]], model: str) -> Tuple[str, int, int, float]:
        start_time = time.time()

        # Calculate input tokens
        input_text = " ".join(msg["content"] for msg in messages)
        tokens_sent = estimate_tokens(input_text)

        request_data = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(request_data).encode('utf-8'),
                headers=headers
            )

            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))
                output_text = result['choices'][0]['message']['content']

                # Calculate output tokens
                tokens_received = estimate_tokens(output_text)
                duration = time.time() - start_time

                return output_text, tokens_sent, tokens_received, duration

        except Exception as e:
            raise RuntimeError(f"API error: {str(e)}")


# Backend factory
def get_backend(backend_type: str = "lmstudio", **kwargs) -> ModelBackend:
    """
    Get the appropriate backend based on type.

    Args:
        backend_type: "ollama", "lmstudio", or "openai-compatible"
        **kwargs: Backend-specific configuration (base_url, api_key, etc.)
    """
    backends = {
        "ollama": OllamaBackend,
        "lmstudio": LMStudioBackend,
        "openai-compatible": OpenAICompatibleBackend,
    }

    if backend_type not in backends:
        raise ValueError(f"Unknown backend: {backend_type}. Choose from: {list(backends.keys())}")

    return backends[backend_type](**kwargs)
