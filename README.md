# Jupiter FAQ Bot

A human-friendly FAQ bot for Jupiter's Help Centre, built with Python, semantic search, and open-source LLMs. The bot scrapes FAQs from Jupiter's help and community pages, preprocesses them, and provides conversational answers via a modern web app frontend.

## Features
- Scrapes and structures FAQs from Jupiter's help and community pages
- Cleans, deduplicates, and categorizes Q&A pairs
- Semantic search using FAISS and sentence-transformers
- Conversational answers using open-source LLMs (via HuggingFace Inference API)
- Modern web app frontend (Streamlit)
- (Bonus) Multilingual support, related query suggestions, and RAG vs. LLM comparison

## Setup

1. Clone the repo and navigate to the project directory:
   ```bash
   git clone <repo-url>
   cd FAQ-Bot
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Project Structure
- `scraper/` - Scripts for scraping and preprocessing FAQ data
- `data/` - Processed and raw FAQ data
- `app.py` - Streamlit web app
- `models/` - Scripts for semantic search and LLM integration
- `README.md` - Project documentation

## To Do
- [ ] Implement FAQ scraping
- [ ] Preprocess and categorize data
- [ ] Build semantic search index
- [ ] Integrate LLM for conversational answers
- [ ] Develop Streamlit frontend
- [ ] Add multilingual support and related query suggestions
- [ ] Compare RAG vs. LLM (accuracy, latency)

---

For more details, see the project plan and code comments.
