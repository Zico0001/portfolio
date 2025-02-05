from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def index_text():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    with open("manual_text.txt", "r") as file:
        text = file.read().split('.')
    embeddings = model.encode(text)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, text

if __name__ == "__main__":
    index, text = index_text()
    faiss.write_index(index, "manual_index.faiss")
    with open("manual_sentences.txt", "w") as file:
        file.write("\n".join(text))
