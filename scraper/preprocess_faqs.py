import json
import os
import re
from collections import OrderedDict

def normalize_text(text):
    # Lowercase, strip, and remove extra spaces
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_faqs(input_path, output_path):
    with open(input_path, 'r') as f:
        faqs = json.load(f)

    # Normalize questions and answers
    for faq in faqs:
        faq['question'] = normalize_text(faq['question'])
        faq['answer'] = faq['answer'].strip()

    # Deduplicate by normalized question
    seen = OrderedDict()
    for faq in faqs:
        q = faq['question']
        if q not in seen:
            seen[q] = faq
    cleaned_faqs = list(seen.values())

    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(cleaned_faqs, f, indent=2, ensure_ascii=False)
    print(f"Preprocessed {len(faqs)} FAQs, deduplicated to {len(cleaned_faqs)}. Saved to {output_path}")

if __name__ == "__main__":
    preprocess_faqs('data/jupiter_faqs_raw.json', 'data/jupiter_faqs_clean.json') 