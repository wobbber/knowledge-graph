import networkx as nx
import spacy
from community import community_louvain
import matplotlib.pyplot as plt
import streamlit as st

class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def process_document(self, text):
        """Processes the document to extract entities and relationships."""
        doc = self.nlp(text)
        for sent in doc.sents:
            entities = [ent for ent in sent.ents]
            for i, ent1 in enumerate(entities):
                for ent2 in entities[i + 1:]:
                    self.graph.add_node(ent1.text, label=ent1.label_)
                    self.graph.add_node(ent2.text, label=ent2.label_)
                    self.graph.add_edge(ent1.text, ent2.text, weight=1)

    def visualize_graph(self):
        """Visualizes the knowledge graph using Streamlit."""
        partition = community_louvain.best_partition(self.graph)

        # Filter nodes based on degree threshold (keeping nodes with a degree >= 2)
        degree_threshold = 2
        filtered_nodes = [node for node, degree in self.graph.degree() if degree >= degree_threshold]
        filtered_graph = self.graph.subgraph(filtered_nodes)

        if not filtered_graph.nodes:
            st.warning("No significant nodes to display in the knowledge graph.")
            return

        # Assign colors for filtered nodes
        cluster_colors = plt.cm.tab20.colors
        node_colors = [
            cluster_colors[partition[node] % len(cluster_colors)] for node in filtered_graph
        ]

        # Adjust the layout for better spacing
        pos = nx.spring_layout(filtered_graph, k=1.5, scale=5, seed=42)

        # Plot the graph
        plt.figure(figsize=(16, 10))
        nx.draw_networkx_nodes(
            filtered_graph, pos, node_size=800, node_color=node_colors, edgecolors="black"
        )
        nx.draw_networkx_edges(filtered_graph, pos, alpha=0.3, edge_color="gray", width=1.5)

        # Draw labels for nodes with the highest degree
        labels = {node: node for node, degree in filtered_graph.degree() if degree >= 3}
        nx.draw_networkx_labels(filtered_graph, pos, labels=labels, font_size=10, font_color="black")

        plt.title("Knowledge Graph", fontsize=20)
        plt.axis("off")

        # Display the plot in Streamlit
        st.pyplot(plt)
