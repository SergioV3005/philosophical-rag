"""
This script retrieves relevant text chunks from a vector store using FAISS and generates an answer to a philosophical question based on the context provided by Friedrich Nietzsche's writings.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve(query, index, chunks, top_k=5):
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb), top_k)
    return [chunks[i] for i in I[0]]

def generate_answer(question, context):
    prompt = f"""You are an expert philosopher specializing in Friedrich Nietzsche's works. Answer the question strictly based on the provided context from Nietzsche's writings:

Context:
{' '.join(context)}

Question:
{question}

Answer:"""

    response = ollama.chat(model='llama3.2', messages=[
        {'role': 'user', 'content': prompt}
    ])
    return response['message']['content'].strip()

if __name__ == '__main__':
    question = input("Enter your philosophical question: ")

    chunks = open('../data/text_chunks.txt', 'r', encoding='utf-8').read().split('\n---\n')
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    index = faiss.read_index('../data/vectorstore.index')

    context = retrieve(question, index, chunks, top_k=5)
    answer = generate_answer(question, context)

    print("\nNietzsche-based Answer:\n")
    print(answer)