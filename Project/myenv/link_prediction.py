import torch
from GNNs import Encoder
from torch_geometric.nn import VGAE
from utils import train_link_prediction_given_epochs
from utils import test_vgae_for_link_prediction
from torch_geometric.transforms import RandomLinkSplit
from graph_dataset_generator import create_dataset

# creating dataset by calling create_dataset function
data, _ = create_dataset()

average_auc_score = 0.0
average_ap_score = 0.0
epochs = 200
for i in range(0, 80): 
    # creating an RandomLinkSplit object to split edges
    # into training, validation, and test sets
    transfrom = RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=False
    )

    train_data, val_data, test_data = transfrom(data)
    # creating an instance of VGAE (Variational Graph Autoencoder) object
    vgae_model = VGAE(Encoder(data.num_features, 16))
    optimizer_vgae = torch.optim.Adam(vgae_model.parameters(), lr=0.01)
    # training VGAE model for 200 epochs
    train_link_prediction_given_epochs(vgae_model, train_data, val_data, optimizer_vgae, epochs, (i+1))
    test_auc, test_ap = test_vgae_for_link_prediction(vgae_model, test_data)
    print(f"Link prediction AUC score in iteration-{i+1} is: {test_auc:.4f}, Link prediction AP score in iteration-{i+1} is: {test_ap:.4f}")
    
    average_auc_score += test_auc
    average_ap_score += test_ap
    
    
average_auc_score /= 80
average_ap_score /= 80   
print(f"Link prediction Average AUC score for test set:{average_auc_score:.4f}, Link prediction Average AP score for test set: {average_ap_score:.4f}")    