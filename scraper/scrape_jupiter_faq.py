import requests
from bs4 import BeautifulSoup
import json
import os
import re

def scrape_jupiter_contact_faq(url):
    """Scrape Q&A pairs from the Jupiter Contact FAQ page."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    faqs = []
    for li in soup.find_all('li', class_=lambda c: c and 'border-black20' in c):
        # Find the question
        question_span = li.find('span', class_=lambda c: c and 'text-black100' in c and 'text-left' in c)
        # Find the answer
        answer_p = li.find('p', class_=lambda c: c and 'text-black60' in c)
        if question_span and answer_p:
            question = question_span.get_text(strip=True)
            # Remove extra whitespace and newlines from answer
            answer = re.sub(r'\s+', ' ', answer_p.get_text(separator=' ', strip=True))
            faqs.append({'question': question, 'answer': answer})
    return faqs

def main():
    url = 'https://jupiter.money/contact/'
    print(f"Scraping: {url}")
    faqs = scrape_jupiter_contact_faq(url)
    os.makedirs('data', exist_ok=True)
    with open('data/jupiter_faqs_raw.json', 'w') as f:
        json.dump(faqs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(faqs)} Q&A pairs to data/jupiter_faqs_raw.json")

if __name__ == "__main__":
    main() 