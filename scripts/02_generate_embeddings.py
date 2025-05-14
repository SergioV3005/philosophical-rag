"""
Generate embeddings for text chunks using a pre-trained model.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_chunks(path):
    with open(path, 'r', encoding='utf-8') as f:
        chunks = [chunk.strip() for chunk in f.read().split('\n---\n') if chunk.strip()]
    return chunks

if __name__ == '__main__':
    chunks = load_chunks('../data/text_chunks.txt')
    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True)

    np.save('../data/embeddings.npy', embeddings)