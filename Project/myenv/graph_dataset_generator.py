import pandas as pd
import networkx as nx
import torch
from sklearn.manifold import TSNE
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils import plot_num_of_nodes_with_degree_info
import numpy as np
import matplotlib.pyplot as plt

# function to create graph dataset for graph learning tasks
def create_dataset(): 
    article_df = pd.read_csv("article_corpus.csv")
    category_counts = article_df["category"].value_counts()

    # display count of different categories
    print(category_counts)

    # loading pretrained SentenceTransformer model (allenai-specter)
    # allenai-specter model can be used to map the titles and abstracts
    # of scientific publications to vector space such that similar 
    # papers are close (paper: https://arxiv.org/abs/2004.07180)
    sentence_transformer = SentenceTransformer("allenai-specter")

    english_article_embeddings = sentence_transformer.encode(article_df["english_abstract"].tolist(), show_progress_bar=True)
    print(english_article_embeddings)
    print(f"Shape of english article embeddings is: {english_article_embeddings.shape}")

    cos_sim_tensor = cos_sim(english_article_embeddings, english_article_embeddings)
    cos_sim_matrix = cos_sim_tensor.numpy()
    print("----COSINE SIMILARITY MATRIX----")
    # print(cos_sim_matrix)   

    """ for paper in range(len(cos_sim_matrix[:1])):
        print(f"Scores for Paper {paper+1}")
        for score in range(len(cos_sim_matrix[paper][:200])):
            print("Environmental Science-->", cos_sim_matrix[paper+623][score+623], "---Sports Science-->",  cos_sim_matrix[paper+623][score+830]) """


    # applying a threshold of 0.73 for connecting
    # nodes that are semantically similar (this threshold is set via analysis)
    np.fill_diagonal(cos_sim_matrix, 0.0)
    edge_index = []
    for i in range(len(article_df)):
        for j in range(i+1, len(article_df)):
            if (cos_sim_matrix[i][j] > 0.73):
                # adding bidirectional edges
                edge_index.append([i ,j])
                edge_index.append([j, i])

    # visualizing high dimensional embeddings using TSNE
    reduced_embeddings = TSNE(n_components=2, perplexity=30, learning_rate="auto", random_state=42).fit_transform(english_article_embeddings)
    print(reduced_embeddings.shape)

    num_of_categories = article_df["category"].nunique()
    cmap = plt.get_cmap("tab10", num_of_categories)
    category_colors = cmap.colors
    categories = article_df["category"].unique().tolist()
    category_labels = article_df["category"].tolist()
    color_category_dict = {categories[i]: category_colors[i] for i in range(0, len(categories))}
    print(color_category_dict)

    for category in categories:
        indices = [i for i, c in enumerate(category_labels) if c == category]
        plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices ,1], c=[color_category_dict[category]], label=category ,alpha=0.8)

    plt.legend()    
    plt.title("Embedding based Graph Nodes (T-SNE)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.savefig("Graph_plots\\Embedding_based_graph_nodes_with_TSNE.png", edgecolor="k", orientation="landscape")

    # creating a networkx graph visualization
    G = nx.Graph()
    for i in range(len(article_df)):
        G.add_node(i, category=category_labels[i])
        
    for edge in edge_index:    
        G.add_edge(edge[0], edge[1])

    # mapping T-SNE coordinates to node positions
    pos = {i: (reduced_embeddings[i, 0], reduced_embeddings[i, 1]) for i in range(len(article_df))}
    # drawing the graph
    for category in categories:
        nodes_in_category = [i for i, c in enumerate(category_labels) if c == category]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes_in_category,
            node_color=[color_category_dict[category]],
            label=category,
            alpha=0.8,
            node_size=20
        )
        
    # drawing edges for graph
    nx.draw_networkx_edges(G, pos, alpha=0.2)    
    plt.title("NetworkX Graph Visualization via Node Categories")
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())
    plt.savefig("Graph_plots\\NetworkX_Graph_Visualization.png", edgecolor="k", orientation="landscape")

    # creating Data object representing graph
    x = torch.tensor(english_article_embeddings, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index.t().contiguous()) 
    # encoding categorical labels
    labels_encoder = LabelEncoder()
    encoded_labels = labels_encoder.fit_transform(article_df["category"])
    y = torch.tensor(encoded_labels, dtype=torch.long)
    print(f"Encoded labels are as follows: {y}")

    num_of_nodes = len(article_df)
    # creating a train-test split for nodes
    train_node_ids, val_plus_test_node_ids = train_test_split(np.arange(num_of_nodes), test_size=0.3, stratify=y, random_state=42)
    val_node_ids, test_node_ids = train_test_split(val_plus_test_node_ids, test_size=0.5, stratify=y[val_plus_test_node_ids])

    # creating training, validation and test set masks
    train_masks = torch.zeros(num_of_nodes, dtype=torch.bool)
    val_masks = torch.zeros(num_of_nodes, dtype=torch.bool)
    test_masks = torch.zeros(num_of_nodes, dtype=torch.bool)

    train_masks[train_node_ids] = True
    val_masks[val_node_ids] = True
    test_masks[test_node_ids] = True

    # assigning training, validation, test masks and labels to Data object
    data.y = y
    data.train_mask = train_masks
    data.val_mask = val_masks
    data.test_mask = test_masks

    # plotting bar plot showing number of nodes having 
    # specific degree
    plot_num_of_nodes_with_degree_info(data.edge_index)

    print(f"Number of features per node is: {data.num_features}")
    print(f"Number of edges in the graph is: {data.num_edges // 2}")

    return data, article_df["category"]
