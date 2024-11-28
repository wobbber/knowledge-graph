import spacy
import networkx as nx
import matplotlib.pyplot as plt
import pdfplumber
import pytesseract
from PIL import Image
from collections import defaultdict
import streamlit as st

import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF document."""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text

    def extract_text_from_image(self, image_path):
        """Extract text from an image."""
        return pytesseract.image_to_string(Image.open(image_path))

    def infer_schema(self, text):
        """Infer schema by identifying entities, relationships, and properties."""
        doc = nlp(text)
        entities = defaultdict(set)
        relationships = []

        # Extract entities
        for ent in doc.ents:
            entities[ent.label_].add(ent.text)

        # Extract relationships (noun chunks with verbs)
        for chunk in doc.noun_chunks:
            if chunk.root.head.pos_ == "VERB":
                relationships.append((chunk.text, chunk.root.head.text, chunk.root.head.i))

        schema = {
            "entities": dict(entities),
            "relationships": relationships,
        }
        return schema

    def build_graph(self, schema):
        """Build knowledge graph from inferred schema."""
        for entity_type, entity_names in schema["entities"].items():
            for name in entity_names:
                self.graph.add_node(name, label=entity_type)

        for subj, rel, obj in schema["relationships"]:
            self.graph.add_edge(subj, obj, relation=rel)

    def visualize_graph(self):
        """Visualize the knowledge graph."""
        pos = nx.spring_layout(self.graph)
        labels = nx.get_edge_attributes(self.graph, "relation")
        nx.draw(self.graph, pos, with_labels=True, node_size=2000, node_color="lightblue")
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels)
        st.pyplot(plt)

    def add_document(self, document_path, file_type="text"):
        """Add and process a new document."""
        if file_type == "pdf":
            text = self.extract_text_from_pdf(document_path)
        elif file_type == "image":
            text = self.extract_text_from_image(document_path)
        else:
            with open(document_path, "r") as file:
                text = file.read()

        schema = self.infer_schema(text)
        self.build_graph(schema)
