import requests
from bs4 import BeautifulSoup
import json
import os

def scrape_faq_page(url):
    """Scrape Q&A pairs from a Jupiter FAQ or help page."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    faqs = []
    # Example: Find all FAQ items (update selectors as needed)
    for item in soup.find_all(['details', 'div'], class_=['faq-item', 'c-topic-list-item']):
        question = item.find(['summary', 'h3', 'h2'])
        answer = item.find(['div', 'p'], class_=['faq-answer', 'c-post__content'])
        if question and answer:
            faqs.append({
                'question': question.get_text(strip=True),
                'answer': answer.get_text(strip=True)
            })
    return faqs

def main():
    urls = [
        'https://jupiter.money/contact/',
        'https://community.jupiter.money/c/help/27',
        'https://community.jupiter.money/faq',
    ]
    all_faqs = []
    for url in urls:
        print(f"Scraping: {url}")
        faqs = scrape_faq_page(url)
        all_faqs.extend(faqs)
    os.makedirs('../data', exist_ok=True)
    with open('../data/jupiter_faqs_raw.json', 'w') as f:
        json.dump(all_faqs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(all_faqs)} Q&A pairs to ../data/jupiter_faqs_raw.json")

if __name__ == "__main__":
    main() 