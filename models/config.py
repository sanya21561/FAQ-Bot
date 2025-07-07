"""
Configuration file for Together API and model settings
"""
import os
# Together API Configuration
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# Model Configuration
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# Alternative models:
# - "meta-llama/Llama-3.1-8B-Instruct" (faster, smaller)
# - "microsoft/DialoGPT-medium" (conversational)
# - "google/gemma-7b-it" (Gemma model)
# - "mistralai/Mistral-7B-Instruct-v0.3" (Mistral model)

# Generation Parameters
MAX_TOKENS = 1500  # Increased for more detailed, stepwise, bullet-pointed answers
TEMPERATURE = 0.7
TOP_P = 0.9

# RAG Configuration
TOP_K_RETRIEVAL = 3  # Use in rag_pipeline.py for top_k
TOP_K_RELATED = 3    # Use in rag_pipeline.py for related questions

# Always import and use these config values in your pipeline and inference code!
