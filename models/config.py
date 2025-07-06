"""
Configuration file for Together API and model settings
"""
import os
# Together API Configuration
# TOGETHER_API_KEY = "e5a55b47c8e2bf0602028d5506d58d19e331d3fb2b59f8c180bcf65a5cb7ba5c"
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# Model Configuration
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# Alternative models:
# - "meta-llama/Llama-3.1-8B-Instruct" (faster, smaller)
# - "meta-llama/Llama-3.1-70B-Instruct" (larger, more capable)
# - "microsoft/DialoGPT-medium" (conversational)
# - "google/gemma-7b-it" (Gemma model)

# Generation Parameters
MAX_TOKENS = 1000
TEMPERATURE = 0.7
TOP_P = 0.9

# RAG Configuration
TOP_K_RETRIEVAL = 1
TOP_K_RELATED = 3 