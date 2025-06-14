import torch
import pandas as pd
from GNNs import GCNBaseline, GraphSAGEBaseline, GATBaseline
from GNNs import GCNCustom, GraphSAGECustom, GATCustom
from torch_geometric.nn.aggr import SoftmaxAggregation, StdAggregation
from graph_dataset_generator import create_dataset
from utils import test_model
from utils import train_model_for_node_clustering
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# creating dataset by calling create_dataset function
data, category_df = create_dataset()
num_of_categories = category_df.nunique()

cmap = plt.get_cmap("tab10", num_of_categories)
category_colors = cmap.colors
categories = category_df.unique().tolist()
category_labels = category_df.tolist()
color_category_dict = {categories[i]: category_colors[i] for i in range(0, len(categories))}

model_name = "GraphSAGECustom6"
average_nmi = 0.0
average_ari = 0.0
epochs = 300
for i in range(0, 80):
    # creating GNN model for node clustering
    gnn_model = GraphSAGECustom(in_channels=data.num_features, hidden_channels=64, out_channels=num_of_categories, num_of_layers=2, dropout_ratio=0.5, aggregation_type=SoftmaxAggregation(t=True))

    # training nodes for clustering
    loss_function = torch.nn.NLLLoss()
    train_model_for_node_clustering(gnn_model, loss_function, model_name, epochs, data, (i+1))
    trained_embeddings, _, _ , _, _, _ = test_model(gnn_model, loss_function, data, model_name, num_of_categories, categories)

    # creating clusters with KMeans
    trained_embeddings = trained_embeddings.numpy()
    clusters = KMeans(n_clusters=5, random_state=42).fit_predict(trained_embeddings)
    # displaying clustering performance with NMI (Normalized Mutual Information score)
    # and ARI (Adjusted Random Index), NMI values close to 1 indicates high correlation
    # values close to 0 indicate low correlation. ARI take values in range -0.5 to 1.0
    # values close to 1 indicate high correlation, whereas values close to 0.0 indicate
    # low correlation, it takes negative values discordant labelings (exactly reverse of what was predicted)
    nmi = normalized_mutual_info_score(data.y.numpy(), clusters)
    ari = adjusted_rand_score(data.y.numpy(), clusters)
    print(f"NMI score for node clustering in iteration-{i+1} is: {nmi:.4f}")
    print(f"ARI score for node clustering in iteration-{i+1} is: {ari:.4f}")
    
    average_nmi += nmi
    average_ari += ari
    
    # visualizing clusters with T-SNE
    reduced_embeddings = TSNE(n_components=2, perplexity=30, learning_rate="auto", random_state=42).fit_transform(trained_embeddings)
    plt.figure(figsize=(8,6))   
    for category in categories:
        indices = [i for i, c in enumerate(category_labels) if c == category]
        plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices ,1], c=[color_category_dict[category]], label=category ,alpha=0.8)

    plt.legend()
    plt.title(f"Iteration-{i+1} t-SNE Visualization for Node Clustering")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.savefig(f"Node_clustering_plots\\{model_name}\\Node_clusters_t_SNE_iteration-{i+1}.png")
    
average_nmi /= 80
average_ari /= 80
print(f"Average NMI score for node clustering is: {average_nmi:.4f}")
print(f"Average ARI score for node clustering is: {average_ari:.4f}")