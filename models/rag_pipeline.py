import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .together_inference import query_together_llm
from .config import TOP_K_RETRIEVAL, TOP_K_RELATED
import re

MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_PATH = 'data/jupiter_faqs_clean.json'
INDEX_PATH = 'models/faq_faiss.index'
EMBEDDINGS_PATH = 'models/faq_embeddings.npy'

def flatten_faqs(faqs):
    flat = []
    def recurse(obj, category=None, subcategory=None):
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and 'question' in item and 'answer' in item:
                    flat.append({
                        'question': item['question'],
                        'answer': item['answer'],
                        'category': category,
                        'subcategory': subcategory
                    })
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if category is None:
                    recurse(v, category=k, subcategory=None)
                else:
                    recurse(v, category=category, subcategory=k)
    recurse(faqs)
    return flat

# Load data and index
with open(DATA_PATH, 'r') as f:
    faqs_nested = json.load(f)
faqs = flatten_faqs(faqs_nested)
questions = [faq['question'] for faq in faqs]
answers = [faq['answer'] for faq in faqs]
categories = [faq.get('category') for faq in faqs]
subcategories = [faq.get('subcategory') for faq in faqs]

index = faiss.read_index(INDEX_PATH)
embeddings = np.load(EMBEDDINGS_PATH)
model = SentenceTransformer(MODEL_NAME)

def retrieve_faq(query, top_k=TOP_K_RETRIEVAL):
    query_emb = model.encode([query]).astype('float32')
    D, I = index.search(query_emb, top_k)
    results = []
    for idx in I[0]:
        results.append({
            'question': questions[idx],
            'answer': answers[idx],
            'category': categories[idx],
            'subcategory': subcategories[idx],
            'index': idx
        })
    return results

def get_related_questions(query, exclude_idx, top_k=TOP_K_RELATED, category=None, subcategory=None):
    query_emb = model.encode([query]).astype('float32')
    D, I = index.search(query_emb, top_k + 10)  # Search more for deduplication/fallback
    related = []
    seen = set()
    user_q_norm = re.sub(r'[^a-z0-9 ]', '', query.lower())
    for idx in I[0]:
        if idx == exclude_idx:
            continue
        q = questions[idx]
        cat = categories[idx]
        subcat = subcategories[idx]
        # Never show the exact same question as the user query
        q_norm = re.sub(r'[^a-z0-9 ]', '', q.lower())
        if q_norm == user_q_norm:
            continue
        # Only show from same category/subcategory if specified
        if category and cat != category:
            continue
        if subcategory and subcat != subcategory:
            continue
        # Deduplicate by question text
        if q_norm in seen:
            continue
        seen.add(q_norm)
        related.append({
            'question': q,
            'category': cat,
            'subcategory': subcat
        })
        if len(related) >= top_k:
            break
    # Fallback: if not enough, fill from any category
    if len(related) < top_k:
        for idx in I[0]:
            if idx == exclude_idx:
                continue
            q = questions[idx]
            q_norm = re.sub(r'[^a-z0-9 ]', '', q.lower())
            if q_norm == user_q_norm or q_norm in seen:
                continue
            cat = categories[idx]
            subcat = subcategories[idx]
            related.append({
                'question': q,
                'category': cat,
                'subcategory': subcat
            })
            seen.add(q_norm)
            if len(related) >= top_k:
                break
    return related

def group_similar_faqs(retrieved):
    # Group FAQs with similar questions (ignoring case/punctuation)
    grouped = []
    seen = set()
    for r in retrieved:
        key = re.sub(r'[^a-z0-9 ]', '', r['question'].lower())
        if key in seen:
            continue
        seen.add(key)
        grouped.append(r)
    return grouped

def rag_answer(user_query, return_prompt=False, top_k=TOP_K_RETRIEVAL):
    # Retrieve top_k FAQs
    retrieved = retrieve_faq(user_query, top_k=top_k)
    retrieved = group_similar_faqs(retrieved)
    # Compose context for LLM
    context = "\n\n".join([
        f"Category: {r['category'] or 'General'} | Subcategory: {r['subcategory'] or '-'}\nQ: {r['question']}\nA: {r['answer']}"
        for r in retrieved
    ])
    prompt = (
        f"User question: {user_query}\n\n"
        f"Relevant FAQs (top {top_k}):\n{context}\n\n"
        "You are a helpful, friendly, and knowledgeable FAQ assistant for Jupiter Money. "
        "If the answer is present in the FAQs above, use and rephrase it in a natural, step-by-step, human-like way. "
        "Use bullet points or numbered steps for clarity when possible. "
        "If the user asks for the meaning of a banking or financial term (e.g., KYC, NEFT, UPI), provide a clear, concise explanation based on your general knowledge. "
        "If you are unsure or the answer is not present, say so politely and suggest contacting support (support@jupiter.money). "
        "Always be truthful, complete, accurate, safe, fluent, and coherent.\n"
        "Start your reply with 'FINAL ANSWER:' and output only the answer."
    )
    llm_response = query_together_llm(prompt)
    # Improved extraction: look for FINAL ANSWER:, then A:/Answer:, then fallback to last paragraph
    def extract_final_answer(llm_response):
        matches = list(re.finditer(r'FINAL ANSWER:\s*', llm_response, re.IGNORECASE))
        if matches:
            if len(matches) >= 2:
                start = matches[1].end()
                after_final = llm_response[start:]
            else:
                start = matches[0].end()
                after_final = llm_response[start:]
                a_match = re.search(r'(A:|Answer:|ANSWER:)\s*', after_final, re.IGNORECASE)
                if a_match:
                    return after_final[a_match.end():].strip()
                if after_final.strip().startswith(".'"):
                    return after_final.strip()[2:].strip()
            cleaned = re.sub(r"^[\s\.'\"-]+", "", after_final)
            return cleaned.strip()
        matches = list(re.finditer(r'(?i)\b(A:|Answer:|ANSWER:)\s*', llm_response))
        if matches:
            last_match = matches[-1].end()
            return llm_response[last_match:].strip()
        paras = [p.strip() for p in llm_response.split('\n') if p.strip()]
        if paras:
            return paras[-1]
        return llm_response.strip()
    final_answer = extract_final_answer(llm_response)
    # Use category/subcategory of top FAQ for related questions
    top_cat = retrieved[0]['category'] if retrieved else None
    top_subcat = retrieved[0]['subcategory'] if retrieved else None
    result = {
        'retrieved_faqs': retrieved,
        'llm_response': final_answer,
        'related_questions': get_related_questions(user_query, exclude_idx=retrieved[0]['index'], top_k=TOP_K_RELATED, category=top_cat, subcategory=top_subcat)
    }
    if return_prompt:
        result['system_prompt'] = prompt
        result['raw_llm_response'] = llm_response
    return result

if __name__ == "__main__":
    user_query = input("Ask a question: ")
    result = rag_answer(user_query)
    print("\n--- Retrieved FAQs ---")
    for r in result['retrieved_faqs']:
        print(f"Category: {r['category'] or 'General'} | Subcategory: {r['subcategory'] or '-'}")
        print(f"Q: {r['question']}")
        print(f"A: {r['answer']}\n")
    print("\n--- Related Questions ---")
    for q in result['related_questions']:
        print(f"Category: {q['category'] or 'General'} | Subcategory: {q['subcategory'] or '-'} | Q: {q['question']}")
    print("\n--- LLM Response ---")
    print(result['llm_response']) 