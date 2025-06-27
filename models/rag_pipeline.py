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
            'answer': answers[idx],
            'index': idx
        })
    return results

def get_related_questions(query, exclude_idx, top_k=3):
    query_emb = model.encode([query]).astype('float32')
    D, I = index.search(query_emb, top_k + 1)  # +1 in case best match is included
    related = []
    for idx in I[0]:
        if idx != exclude_idx and len(related) < top_k:
            related.append(questions[idx])
    return related

def rag_answer(user_query, return_prompt=False):
    # Retrieve best FAQ
    retrieved = retrieve_faq(user_query, top_k=1)[0]
    # Related questions (top 3, excluding best match)
    related_questions = get_related_questions(user_query, exclude_idx=retrieved['index'], top_k=3)
    # Compose prompt for LLM
    prompt = f"User question: {user_query}\n\nRelevant FAQ:\nQ: {retrieved['question']}\nA: {retrieved['answer']}\n\nIf the FAQ is relevant, answer in a friendly way. If not, try to answer or say you don't know."
    llm_response = query_huggingface_llm(prompt)
    # Extract only the final answer after the last 'A:' or 'Answer:' (case-insensitive)
    final_answer = llm_response
    matches = list(re.finditer(r'(?i)\b(A:|Answer:)\s*', llm_response))
    if matches:
        last_match = matches[-1].end()
        final_answer = llm_response[last_match:].strip()
    if not final_answer:
        final_answer = llm_response.strip()
    result = {
        'retrieved_faq': retrieved,
        'llm_response': final_answer,
        'related_questions': related_questions
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
    print("\n--- Related Questions ---")
    for q in result['related_questions']:
        print(q)
    print("\n--- LLM Response ---")
    print(result['llm_response']) 