import streamlit as st
from knowledge_graph import KnowledgeGraphBuilder
import PyPDF2

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# App title and introduction
st.title("Automated Knowledge Graph Builder")
st.write("Upload a structured PDF document to generate and visualize a knowledge graph.")

# File upload widget
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

if uploaded_file:
    # Extract text from the uploaded PDF
    text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Document Content", text, height=300)

    # Initialize and process the document
    st.write("### Knowledge Graph Visualization")
    kg_builder = KnowledgeGraphBuilder()
    kg_builder.process_document(text)

    # Visualize the knowledge graph
    kg_builder.visualize_graph()
