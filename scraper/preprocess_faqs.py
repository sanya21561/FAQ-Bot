import json
import os
import re
from collections import OrderedDict
from datasketch import MinHash

def normalize_text(text):
    # Lowercase, strip, and remove extra spaces
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def get_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for token in text.split():
        m.update(token.encode('utf8'))
    return m

def preprocess_faqs(input_path, output_path, minhash_threshold=0.8):
    with open(input_path, 'r') as f:
        faqs = json.load(f)

    # Normalize questions and answers
    for faq in faqs:
        faq['question'] = normalize_text(faq['question'])
        faq['answer'] = faq['answer'].strip()

    # Deduplicate by MinHash similarity of normalized question
    minhashes = []
    unique_faqs = []
    for faq in faqs:
        mh = get_minhash(faq['question'])
        is_duplicate = False
        for existing_mh in minhashes:
            if mh.jaccard(existing_mh) >= minhash_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            minhashes.append(mh)
            unique_faqs.append(faq)

    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(unique_faqs, f, indent=2, ensure_ascii=False)
    print(f"Preprocessed {len(faqs)} FAQs, deduplicated to {len(unique_faqs)}. Saved to {output_path}")

if __name__ == "__main__":
    preprocess_faqs('data/jupiter_faqs_raw.json', 'data/jupiter_faqs_clean.json') 