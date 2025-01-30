import streamlit as st
import fitz  # PyMuPDF
from io import BytesIO
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

# Load Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Text extraction function using fitz (PyMuPDF)
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=BytesIO(uploaded_file.read()))  # Use the in-memory file
    text = ""
    for page_num in range(doc.page_count):  # Loop through all pages
        page = doc.load_page(page_num)  # Load each page
        text += page.get_text()  # Extract text from each page
    return text

# Text preprocessing to chunk the document
def preprocess_text(text, chunk_size=300):
    text = re.sub(r'\s+', ' ', text)
    sentences = text.split('.')
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < chunk_size:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# Semantic search to find the most relevant chunks
def semantic_search(query, chunks):
    query_embedding = model.encode([query])
    chunk_embeddings = model.encode(chunks)
    similarities = cosine_similarity(query_embedding, chunk_embeddings).flatten()
    sorted_indices = np.argsort(-similarities)
    return [chunks[i] for i in sorted_indices[:5]]

# Streamlit App UI
st.title("📄 Research Paper Q&A with AI")

uploaded_file = st.file_uploader("📤 Upload a Research Paper (PDF format)", type="pdf")

if uploaded_file is not None:
    with st.spinner("⏳ Extracting text from PDF..."):
        try:
            text = extract_text_from_pdf(uploaded_file)  # Extract text from PDF
            st.success("✅ Text extracted and processed!")
            chunks = preprocess_text(text)  # Preprocess the text into chunks
        except Exception as e:
            st.error(f"Error: {e}")

    query = st.text_input("Enter your query:")

    if query:
        with st.spinner("🔍 Searching for relevant content..."):
            relevant_chunks = semantic_search(query, chunks)  # Perform semantic search
            context = " ".join(relevant_chunks)
        
        st.subheader("Answer")
        st.write(context)  # Display the top relevant chunks
