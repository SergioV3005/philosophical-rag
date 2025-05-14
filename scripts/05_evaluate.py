""""
The idea is to evaluate the generated answer by comparison to an expert answer."""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(answer, reference):
    answer_emb = model.encode([answer])
    reference_emb = model.encode([reference])
    return cosine_similarity(answer_emb, reference_emb)[0][0]

if __name__ == '__main__':
    generated_answer = input("Paste generated answer: ")
    reference_answer = input("Paste reference/expert answer: ")

    similarity_score = semantic_similarity(generated_answer, reference_answer)

    print(f"\nSemantic similarity score: {similarity_score:.4f}")