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
    # Compose prompt for LLM
    prompt = (
        f"User question: {user_query}\n\n"
        f"Relevant FAQ:\nQ: {retrieved['question']}\nA: {retrieved['answer']}\n\n"
        "If the FAQ is relevant, answer in a friendly way. If not, try to answer or say you don't know. "
        "If the user is greeting (e.g., 'hello', 'hi', 'how are you'), greet back and ask if they have any queries about Jupiter Money. Do not answer with unrelated FAQ content. "
        "Start your answer with 'FINAL ANSWER:'."
    )
    llm_response = query_huggingface_llm(prompt)
    # Improved extraction: look for FINAL ANSWER:, then A:/Answer:, then fallback to last paragraph
    def extract_final_answer(llm_response):
        # Try FINAL ANSWER: (use last occurrence)
        matches = list(re.finditer(r'FINAL ANSWER:\s*', llm_response, re.IGNORECASE))
        if matches:
            last_match = matches[-1].end()
            after_final = llm_response[last_match:]
            # If only one FINAL ANSWER:, check for A: or Answer: at the start
            a_match = re.match(r"[\s\.'\"]*(A:|Answer:)\s*", after_final, re.IGNORECASE)
            if a_match:
                return after_final[a_match.end():].strip()
            # If starts with stray .', skip first two chars
            if after_final.strip().startswith(".'"):
                return after_final.strip()[2:].strip()
            return after_final.strip()
        # Try A: or Answer:
        matches = list(re.finditer(r'(?i)\b(A:|Answer:)\s*', llm_response))
        if matches:
            last_match = matches[-1].end()
            return llm_response[last_match:].strip()
        # Fallback: last paragraph
        paras = [p.strip() for p in llm_response.split('\n') if p.strip()]
        if paras:
            return paras[-1]
        return llm_response.strip()
    final_answer = extract_final_answer(llm_response)
    result = {
        'retrieved_faq': retrieved,
        'llm_response': final_answer,
        'related_questions': get_related_questions(user_query, exclude_idx=retrieved['index'], top_k=3)
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