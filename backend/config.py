import os

NUM_VARIANTS = 4
NUM_VARIANTS_VIDEO = 2

# LLM-related
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)

# Chat Completions compatible models (optional, falls back to OpenAI keys)
OPENAI_CHAT_API_KEY = os.environ.get("OPENAI_CHAT_API_KEY", None)
OPENAI_CHAT_BASE_URL = os.environ.get("OPENAI_CHAT_BASE_URL", None)
OPENAI_CHAT_MODELS = os.environ.get("OPENAI_CHAT_MODELS", "qwen3.6-plus").split(",")

# Image generation (optional)
REPLICATE_API_KEY = os.environ.get("REPLICATE_API_KEY", None)

# Debugging-related
IS_DEBUG_ENABLED = bool(os.environ.get("IS_DEBUG_ENABLED", False))
DEBUG_DIR = os.environ.get("DEBUG_DIR", "")

# Set to True when running in production (on the hosted version)
# Used as a feature flag to enable or disable certain features
IS_PROD = os.environ.get("IS_PROD", False)
