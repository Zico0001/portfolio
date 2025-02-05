import streamlit as st
import requests
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Function to download PDF from GitHub
def download_pdf_from_github(url, save_path):
    response = requests.get(url)
    with open(save_path, "wb") as file:
        file.write(response.content)

# Function to extract text from PDF using PyPDF2
def extract_text_from_pdf(pdf_path):
    import PyPDF2
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfFileReader(file)
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extract_text()
    return text

# Function to index text using FAISS
def index_text(text):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sentences = text.split('.')
    embeddings = model.encode(sentences)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, sentences

# Streamlit app
st.title("PDF Manual Chatbot")

pdf_url = "https://github.com/Zico0001/Manual_AI_Expert/blob/main/manual.pdf"
pdf_path = "manual.pdf"

# Download PDF
download_pdf_from_github(pdf_url, pdf_path)

# Extract text
text = extract_text_from_pdf(pdf_path)

# Index text
index, sentences = index_text(text)

@st.cache(allow_output_mutation=True)
def load_model():
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')
    return generator

generator = load_model()

def search_manual(query, index, sentences):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    return sentences[I[0][0]]

user_query = st.text_input("Ask me about the manual:")

if user_query:
    manual_response = search_manual(user_query, index, sentences)
    response = generator(f"Based on the manual: {manual_response}\nUser: {user_query}\nBot:", max_length=100, num_return_sequences=1)
    st.write(f"Bot: {response[0]['generated_text'].strip()}")

