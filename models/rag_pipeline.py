import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llm_inference import query_huggingface_llm

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

def rag_answer(user_query):
    # Retrieve best FAQ
    retrieved = retrieve_faq(user_query, top_k=1)[0]
    # Compose prompt for LLM
    prompt = f"User question: {user_query}\n\nRelevant FAQ:\nQ: {retrieved['question']}\nA: {retrieved['answer']}\n\nIf the FAQ is relevant, answer in a friendly way. If not, try to answer or say you don't know."
    llm_response = query_huggingface_llm(prompt)
    return {
        'retrieved_faq': retrieved,
        'llm_response': llm_response
    }

if __name__ == "__main__":
    user_query = input("Ask a question: ")
    result = rag_answer(user_query)
    print("\n--- Retrieved FAQ ---")
    print(f"Q: {result['retrieved_faq']['question']}")
    print(f"A: {result['retrieved_faq']['answer']}")
    print("\n--- LLM Response ---")
    print(result['llm_response']) 