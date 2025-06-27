import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_PATH = 'data/jupiter_faqs_clean.json'
INDEX_PATH = 'models/faq_faiss.index'
EMBEDDINGS_PATH = 'models/faq_embeddings.npy'

os.makedirs('models', exist_ok=True)

# Load cleaned FAQ data
with open(DATA_PATH, 'r') as f:
    faqs = json.load(f)
questions = [faq['question'] for faq in faqs]

# Generate embeddings
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(questions, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)
np.save(EMBEDDINGS_PATH, embeddings)
print(f"FAISS index and embeddings saved. {len(questions)} questions indexed.") 