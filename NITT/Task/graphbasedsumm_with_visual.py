import nltk
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Input text
text = """
India is the seventh-largest country by land area and the most populous democracy in the world.
It is bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast.
India shares land borders with Pakistan to the northwest, China and Nepal to the north, Bhutan to the northeast, and Bangladesh and Myanmar to the east.
In the Indian Ocean, India is in the vicinity of Sri Lanka and the Maldives.
India's Andaman and Nicobar Islands share a maritime border with Thailand and Indonesia.
India has a rich history and is known for its cultural diversity.
New Delhi is the capital of India.
India is one of the world's fastest-growing major economies.
Technology and innovation are also increasing rapidly in India.
The country has made significant progress in space technology.
"""

# Step 2: Preprocessing
nltk.download('punkt')
sentences = nltk.sent_tokenize(text)

# Step 3: Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sentences)
similarity_matrix = cosine_similarity(tfidf_matrix)

# Step 4: Graph creation
nx_graph = nx.from_numpy_array(similarity_matrix)
scores = nx.pagerank(nx_graph)

# Step 5: Ranking
ranked = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
summary_len = 3
summary_nodes = [index for _, index, _ in ranked[:summary_len]]

# Step 6: Display summary
print("\n📘 Summary:\n")
for _, _, sentence in ranked[:summary_len]:
    print("-", sentence)

# Step 7: Graph Visualization
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(nx_graph, seed=42)

# Sentence labels (wrapped to max 60 chars)
labels = {i: s if len(s) < 60 else s[:60] + '...' for i, s in enumerate(sentences)}

# Node colors: highlight summary sentences
node_colors = ['lightgreen' if i in summary_nodes else 'lightblue' for i in range(len(sentences))]

# Edge weights = similarity scores
edges = nx_graph.edges(data=True)
weights = [similarity_matrix[u][v] for u, v in nx_graph.edges()]

# Draw components
nx.draw_networkx_nodes(nx_graph, pos, node_color=node_colors, node_size=1200, edgecolors='black')
nx.draw_networkx_edges(nx_graph, pos, width=[w * 5 for w in weights], alpha=0.5)
nx.draw_networkx_labels(nx_graph, pos, labels, font_size=8)

# Edge weight labels
edge_labels = {(u, v): f"{similarity_matrix[u][v]:.2f}" for u, v in nx_graph.edges()}
nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_size=6)

# Title and legend
plt.title("Enhanced Sentence Similarity Graph (Graph-Based Summarization)", fontsize=14)
summary_patch = plt.Line2D([0], [0], marker='o', color='w', label='Top Summary Sentence',
                          markerfacecolor='lightgreen', markersize=10, markeredgecolor='black')
normal_patch = plt.Line2D([0], [0], marker='o', color='w', label='Other Sentence',
                          markerfacecolor='lightblue', markersize=10, markeredgecolor='black')
plt.legend(handles=[summary_patch, normal_patch], loc='best')

plt.axis('off')
plt.tight_layout()
plt.show()
