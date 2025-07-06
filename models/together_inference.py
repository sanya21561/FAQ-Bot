import os
import requests
import json
from .config import TOGETHER_API_KEY, TOGETHER_API_URL, DEFAULT_MODEL, MAX_TOKENS, TEMPERATURE, TOP_P

def query_together_llm(prompt, model=DEFAULT_MODEL):
    """
    Query Together AI's API for LLM inference
    
    Args:
        prompt (str): The input prompt for the model
        model (str): The model to use (default: Llama-3.3-70B-Instruct-Turbo)
    
    Returns:
        str: The generated response from the model
    """
    api_url = TOGETHER_API_URL
    
    # Use the API key from config
    api_key = TOGETHER_API_KEY
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract the response content
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            raise Exception("No response content found in API response")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Together API request failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse API response: {str(e)}")
    except Exception as e:
        raise Exception(f"Together API error: {str(e)}")

def query_together_llm_with_system_prompt(system_prompt, user_prompt, model=DEFAULT_MODEL):
    """
    Query Together AI's API with a system prompt and user prompt
    
    Args:
        system_prompt (str): The system prompt to set context
        user_prompt (str): The user's question/prompt
        model (str): The model to use
    
    Returns:
        str: The generated response from the model
    """
    api_url = TOGETHER_API_URL
    
    # Use the API key from config
    api_key = TOGETHER_API_KEY
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract the response content
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            raise Exception("No response content found in API response")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Together API request failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse API response: {str(e)}")
    except Exception as e:
        raise Exception(f"Together API error: {str(e)}")

if __name__ == "__main__":
    # Test the API
    test_prompt = "What is Jupiter Money?"
    print("Testing Together API...")
    try:
        response = query_together_llm(test_prompt)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}") 