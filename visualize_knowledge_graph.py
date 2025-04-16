# visualize_knowledge_graph.py

import networkx as nx
import matplotlib.pyplot as plt

# Example input format (you'll replace this with real data)
concepts = ['transformers', 'attention', 'language models']
papers = [
    {"title": "Attention Is All You Need", "paperId": "p1", "concepts": ["attention", "transformers"]},
    {"title": "BERT: Pre-training of Deep Bidirectional Transformers", "paperId": "p2", "concepts": ["language models", "transformers"]},
    {"title": "A Survey of Transformers", "paperId": "p3", "concepts": ["transformers"]}
]
citation_edges = [("p1", "p2"), ("p2", "p3")]  # p1 cites p2, p2 cites p3

# Create a directed graph
G = nx.DiGraph()

# Add concept nodes
for concept in concepts:
    G.add_node(concept, type="concept")

# Add paper nodes and edges from concepts to papers
for paper in papers:
    G.add_node(paper["paperId"], label=paper["title"], type="paper")
    for concept in paper.get("concepts", []):
        if concept in concepts:
            G.add_edge(concept, paper["paperId"])

# Add citation edges between papers
for source, target in citation_edges:
    if G.has_node(source) and G.has_node(target):
        G.add_edge(source, target)

# Visualization
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(12, 8))

# Draw nodes by type
concept_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "concept"]
paper_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "paper"]

nx.draw_networkx_nodes(G, pos, nodelist=concept_nodes, node_color="skyblue", node_shape="o", label="Concepts")
nx.draw_networkx_nodes(G, pos, nodelist=paper_nodes, node_color="lightgreen", node_shape="s", label="Papers")
nx.draw_networkx_edges(G, pos, arrows=True)

# Labels: use paper titles if available
labels = {
    n: (G.nodes[n].get("label") if G.nodes[n].get("type") == "paper" else n)
    for n in G.nodes
}
nx.draw_networkx_labels(G, pos, labels, font_size=9)

plt.title("Knowledge Graph: Concepts and Citations")
plt.legend()
plt.axis("off")
plt.tight_layout()
plt.show()
