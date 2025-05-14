"""
Script to generate text chunks from a PDF file.
"""

import fitz

def parse_pdf(path, chunk_size=300):
    doc = fitz.open(path)
    texts = []
    full_text = ''

    for page in doc:
        full_text += page.get_text()

    words = full_text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

if __name__ == '__main__':
    chunks = parse_pdf('../data/thus spoke zarathustra.pdf')
    with open('../data/text_chunks.txt', 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + '\n---\n')