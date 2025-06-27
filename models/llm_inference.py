import os
import requests

def query_huggingface_llm(prompt, model="mistralai/Mistral-7B-Instruct-v0.3"):
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}", "Content-Type": "application/json"}
    payload = {"inputs": prompt}
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        # HuggingFace returns a list of dicts with 'generated_text'
        if isinstance(result, list) and 'generated_text' in result[0]:
            return result[0]['generated_text']
        # Some models return 'data' or 'text'
        return result
    else:
        raise Exception(f"HuggingFace API error: {response.status_code} {response.text}")

if __name__ == "__main__":
    prompt = "What is Jupiter Money?"
    print(query_huggingface_llm(prompt)) 