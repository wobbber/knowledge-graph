import streamlit as st
from knowledge_graph import KnowledgeGraphBuilder

# Initialize Knowledge Graph
kg_builder = KnowledgeGraphBuilder()

st.title("Automated Knowledge Graph Builder")
uploaded_file = st.file_uploader("Upload a Document", type=["pdf", "txt", "png"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]
    if file_type == "pdf":
        file_type = "pdf"
    elif file_type in ["png", "jpg", "jpeg"]:
        file_type = "image"
    else:
        file_type = "text"

    with open(f"data/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.read())

    st.write(f"Processing {uploaded_file.name}...")
    kg_builder.add_document(f"data/{uploaded_file.name}", file_type=file_type)

    st.write("### Knowledge Graph Visualization:")
    kg_builder.visualize_graph()
