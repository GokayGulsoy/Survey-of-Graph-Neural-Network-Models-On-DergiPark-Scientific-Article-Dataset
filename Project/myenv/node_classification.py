from torch_geometric.nn.aggr import SoftmaxAggregation
from torch_geometric.nn.aggr import StdAggregation
from GNNs import GCNBaseline, GraphSAGEBaseline, GATBaseline
from GNNs import GCNCustom, GraphSAGECustom, GATCustom
from utils import train_node_classification_given_epochs
from graph_dataset_generator import create_dataset

# creating dataset by calling create_dataset function
data, category_df = create_dataset()
categories = category_df.unique().tolist()
num_of_categories = category_df.nunique()

# training model for 180 epochs
epochs = 180

average_test_loss = 0.0
average_test_accuracy = 0.0
average_f1_score = 0.0
average_precision = 0.0
average_recall = 0.0

train_losses = []
val_losses = []
gnn_model_name = "GCNCustom6"
for i in range(80):    
    # creating instances of gnn model for training
    gnn_model = GCNCustom(in_channels=data.num_features, hidden_channels=64 ,out_channels=num_of_categories, num_of_layers=2, dropout_ratio=0.5)
    
    _, test_loss, test_accuracy, f1_score, precision, recall = train_node_classification_given_epochs(data, gnn_model, gnn_model_name, epochs, categories, train_losses, val_losses, (i+1))

    average_test_loss += test_loss
    average_test_accuracy += test_accuracy
    average_f1_score += f1_score
    average_precision += precision
    average_recall += recall

# displaying averaged metrics 
print(f"Average test loss for {gnn_model_name} is: {(average_test_loss / 80):.4f}")
print(f"Average test accuracy for {gnn_model_name} is: {(average_test_accuracy / 80):.4f}")
print(f"Average F1 score for {gnn_model_name} is: {(average_f1_score / 80):.4f}")
print(f"Average precision for {gnn_model_name} is: {(average_precision / 80):.4f}")
print(f"Average recall for {gnn_model_name} is: {(average_recall / 80):.4f}")
