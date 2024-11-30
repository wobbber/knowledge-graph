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
                    # Add an edge between entities, incrementing weight for repeated connections
                    if self.graph.has_edge(ent1.text, ent2.text):
                        self.graph[ent1.text][ent2.text]['weight'] += 1
                    else:
                        self.graph.add_edge(ent1.text, ent2.text, weight=1)

    def visualize_graph(self):
        """Visualizes the knowledge graph using Streamlit, focusing on the most important points."""
        # Calculate node degrees and edge weights to filter
        degree_threshold = 4  # Minimum degree to consider a node important
        weight_threshold = 2  # Minimum edge weight for connections to be shown

        # Filter nodes by degree: Select top nodes with highest degree
        sorted_nodes = sorted(self.graph.degree, key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, _ in sorted_nodes[:20]]  # Top 20 nodes

        # Create a new graph with the top nodes
        filtered_graph = nx.Graph(self.graph.subgraph(top_nodes))

        # Remove edges below the weight threshold
        edges_to_remove = [
            (u, v) for u, v, w in filtered_graph.edges(data=True) if w["weight"] < weight_threshold
        ]
        filtered_graph.remove_edges_from(edges_to_remove)

        # Warn if the graph is empty after filtering
        if not filtered_graph.nodes:
            st.warning("No significant nodes to display in the knowledge graph.")
            return

        # Assign cluster-based colors
        partition = community_louvain.best_partition(filtered_graph)
        cluster_colors = plt.cm.tab20.colors
        node_colors = [
            cluster_colors[partition[node] % len(cluster_colors)] for node in filtered_graph
        ]

        # Adjust layout for better visualization
        pos = nx.spring_layout(filtered_graph, k=1.5, scale=5, seed=42)

        # Plot the filtered graph
        plt.figure(figsize=(14, 8))
        nx.draw_networkx_nodes(
            filtered_graph, pos, node_size=800, node_color=node_colors, edgecolors="black"
        )
        nx.draw_networkx_edges(filtered_graph, pos, alpha=0.5, edge_color="gray", width=1.5)

        # Add labels for nodes
        labels = {node: node for node in filtered_graph.nodes}
        nx.draw_networkx_labels(filtered_graph, pos, labels=labels, font_size=10, font_color="black")

        plt.title("Filtered Knowledge Graph (Important Points Only)", fontsize=18)
        plt.axis("off")

        # Render the plot in Streamlit
        st.pyplot(plt)
        plt.close()  # Close the figure to free memory
