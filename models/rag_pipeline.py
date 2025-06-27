import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .llm_inference import query_huggingface_llm
import re

MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_PATH = 'data/jupiter_faqs_clean.json'
INDEX_PATH = 'models/faq_faiss.index'
EMBEDDINGS_PATH = 'models/faq_embeddings.npy'

# Load data and index
with open(DATA_PATH, 'r') as f:
    faqs = json.load(f)
questions = [faq['question'] for faq in faqs]
answers = [faq['answer'] for faq in faqs]

index = faiss.read_index(INDEX_PATH)
embeddings = np.load(EMBEDDINGS_PATH)
model = SentenceTransformer(MODEL_NAME)

def retrieve_faq(query, top_k=1):
    query_emb = model.encode([query]).astype('float32')
    D, I = index.search(query_emb, top_k)
    results = []
    for idx in I[0]:
        results.append({
            'question': questions[idx],
            'answer': answers[idx]
        })
    return results

def rag_answer(user_query, return_prompt=False):
    # Retrieve best FAQ
    retrieved = retrieve_faq(user_query, top_k=1)[0]
    # Compose prompt for LLM
    prompt = f"User question: {user_query}\n\nRelevant FAQ:\nQ: {retrieved['question']}\nA: {retrieved['answer']}\n\nIf the FAQ is relevant, answer in a friendly way. If not, try to answer or say you don't know."
    llm_response = query_huggingface_llm(prompt)
    # Extract only the final answer after the last 'A:'
    final_answer = llm_response
    # Try to extract after the last 'A:'
    matches = list(re.finditer(r'\bA:\s*', llm_response))
    if matches:
        last_a = matches[-1].end()
        final_answer = llm_response[last_a:].strip()
    # If nothing after 'A:', fallback to the whole response
    if not final_answer:
        final_answer = llm_response.strip()
    result = {
        'retrieved_faq': retrieved,
        'llm_response': final_answer
    }
    if return_prompt:
        result['system_prompt'] = prompt
        result['raw_llm_response'] = llm_response
    return result

if __name__ == "__main__":
    user_query = input("Ask a question: ")
    result = rag_answer(user_query)
    print("\n--- Retrieved FAQ ---")
    print(f"Q: {result['retrieved_faq']['question']}")
    print(f"A: {result['retrieved_faq']['answer']}")
    print("\n--- LLM Response ---")
    print(result['llm_response']) 