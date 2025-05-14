"""
Generate the vector store from the embeddings.
"""

import faiss
import numpy as np

if __name__ == '__main__':
    embeddings = np.load('../data/embeddings.npy')
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, '../data/vectorstore.index')