import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import degree
from sklearn.metrics import classification_report
from collections import Counter 
from torcheval.metrics.functional import multiclass_f1_score, multiclass_precision, multiclass_recall

# function to train model, it returns training loss
def train_model(model, optimizer, loss_function, data):
    model.train()
    optimizer.zero_grad()
    trained_embeddings = model(data.x, data.edge_index)
    loss = loss_function(trained_embeddings[data.train_mask], data.y[data.train_mask].squeeze())
    loss.backward()
    optimizer.step()

    return loss.item()

# function to validate model, it returns validation loss
def validate_model(model, loss_function, data):
    model.eval()
    with torch.no_grad():
        trained_embeddings = model(data.x, data.edge_index)
        val_loss = loss_function(trained_embeddings[data.val_mask], data.y[data.val_mask])
    
    return val_loss.item()

# function to test model for node classification
def test_model(model, loss_function, data, model_name, num_of_classes, categories):
    model.eval()
    with torch.no_grad():
        test_embeddings = model(data.x, data.edge_index)
        test_loss = loss_function(test_embeddings[data.test_mask], data.y[data.test_mask].squeeze())        
        predictions = test_embeddings[data.test_mask].argmax(dim=1)
        
        # computing accuracy, F1 score, precision and recall metrics
        test_accuracy = (predictions == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        f1_score = multiclass_f1_score(predictions, data.y[data.test_mask], num_classes=num_of_classes)
        precision = multiclass_precision(predictions, data.y[data.test_mask], num_classes=num_of_classes)
        recall = multiclass_recall(predictions, data.y[data.test_mask], num_classes=num_of_classes)
        
        # displaying computed metrics
        print(f"Test Loss for {model_name} model is: {test_loss.item():.4f}")
        print(f"Test Accuracy for {model_name} model is: {test_accuracy:.4f}")
        class_report = classification_report(data.y[data.test_mask], predictions, target_names=categories)
        print(class_report)
        
        return test_embeddings, test_loss, test_accuracy, f1_score, precision, recall 
    
    
# function to generate training and validation loss curves for given model
def generate_loss_curves(training_losses, val_losses, epochs, model_name, output_path, iteration_num):
    training_losses = np.array(training_losses)
    val_losses = np.array(val_losses)
    epochs = np.arange(epochs)
    
    plt.figure()
    plt.title(model_name +  " Iteration -" + str(iteration_num) + " Training and Validation Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.plot(epochs, training_losses, label=model_name + "Training Loss")
    plt.plot(epochs, val_losses, label=model_name + " Validation Loss")
    plt.legend()
    plt.savefig(f"{output_path}\\{model_name}\\{model_name}-{iteration_num}_training_and_validation_loss_curves.png")
    
    
# function to train given GCN and GraphSAGE models for given number of epochs
def train_node_classification_given_epochs(data, gnn_model, gnn_model_name, epochs, categories, average_train_losses, average_val_losses, iteration_num):    
    # creating lists to accumulate training and validation losses 
    train_losses = []
    val_losses = []
    
    # creating instances of optimizers and NLLLoss (loss function) 
    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
    
    for epoch in range(0, epochs): 
        loss = train_model(gnn_model, optimizer, loss_function, data)
        train_losses.append(loss)
        val_losses.append(validate_model(gnn_model, loss_function, data)) 
    
        if epoch % 10 == 0: 
            print(f"Epoch {epoch}, {gnn_model_name} loss: {loss:.4f}, {gnn_model_name} Validation Loss: {val_losses[-1]:.4f}")

    # append training losses and validation losses to average training losses 
    # and average validation losses lists
    average_train_losses.append(train_losses)
    average_val_losses.append(val_losses)
    # generating loss curve for given GNN model
    generate_loss_curves(train_losses, val_losses, epochs, gnn_model_name, "Node_classification_plots", iteration_num)

    # testing the model with GCN and GraphSAGE 
    return test_model(gnn_model, loss_function, data, gnn_model_name, len(categories), categories)


# functin to train model for node clustering task
def train_model_for_node_clustering(model, loss_function, gnn_model_name, epochs, data, iteration_num):
    # creating lists to accumulate training and validation losses 
    train_losses = []
    val_losses = []
    # creating instances of optimizer and NLLLoss (loss function) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(0, epochs):
        loss = train_model(model, optimizer, loss_function, data)
        train_losses.append(loss)
        val_losses.append(validate_model(model, loss_function, data)) 

        if epoch % 10 == 0: 
            print(f"Epoch {epoch}, {gnn_model_name} loss: {loss:.4f}, {gnn_model_name} Validation Loss: {val_losses[-1]:.4f}")

    # generating loss curves for GCN and GraphSAGE models under plots directory
    generate_loss_curves(train_losses, val_losses, epochs, gnn_model_name, "Node_clustering_plots", iteration_num)
    
    
# function to plot number of nodes having specific node degree
def plot_num_of_nodes_with_degree_info(edge_index):
    degrees = degree(edge_index[0]).numpy()
    node_degrees_and_numbers = Counter(degrees)
    plt.figure()
    plt.xlabel("Node degree")
    plt.ylabel("Number of nodes")
    plt.bar(node_degrees_and_numbers.keys(), node_degrees_and_numbers.values())
    plt.savefig("Graph_plots\\Node_Numbers_by_Degree")

# function to train VGAE (Variational Graph Autoencoder) model for link prediction
def train_vgae_for_link_prediction(vgae_model, training_data, optimizer):
    vgae_model.train()
    optimizer.zero_grad()
    encoded_embeddings = vgae_model.encode(training_data.x, training_data.edge_index)
    loss = vgae_model.recon_loss(encoded_embeddings, training_data.pos_edge_label_index)
    loss = loss + (1 / training_data.num_nodes) * vgae_model.kl_loss()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# function to test VGAE (Variational Graph Autoencoder) model for link prediction
def test_vgae_for_link_prediction(vgae_model, test_data):
    vgae_model.eval()
    with torch.no_grad():
        test_embeddings = vgae_model.encode(test_data.x, test_data.edge_index)
        # obtaining AUC (area under curve and average precision metrics)
        auc, ap = vgae_model.test(test_embeddings, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
        
    return auc, ap


# function to plot AUC and AP curves for training and validation datasets
def generate_plots_for_link_prediction(val_auc_scores, val_ap_scores, epochs, iteration_num):
    val_auc_scores = np.array(val_ap_scores)
    val_ap_scores = np.array(val_ap_scores)
    
    plt.figure()
    plt.title(f"VGAE Iteration {iteration_num} Link Prediction AUC Scores")
    plt.xlabel("Epochs")
    plt.ylabel("AUC Score")
    plt.plot(np.arange(epochs), val_auc_scores, label= "Validation AUC Score")
    plt.legend()
    plt.savefig(f"Link_prediction_plots\\VGAE-{iteration_num}_validation_AUC_curve.png")
    
    plt.figure()
    plt.title(f"VGAE Iteration {iteration_num} Link Prediction AP Scores")
    plt.xlabel("Epochs")
    plt.ylabel("AP Score")
    plt.plot(np.arange(epochs), val_ap_scores, label= "Validation AP Score")
    plt.legend()
    plt.savefig(f"Link_prediction_plots\\VGAE-{iteration_num}_validation_AP_curve.png")
    
    
# function to train vgae model for link prediction task
def train_link_prediction_given_epochs(vgae_model, training_data, val_data, optimizer, epochs, iteration_num):
    val_auc_scores = []
    val_ap_scores = []
    
    for epoch in range(0, epochs):
        # run for given epochs and save auc ap values from validation as well
        train_loss = train_vgae_for_link_prediction(vgae_model, training_data, optimizer)
        vgae_model.eval()
        with torch.no_grad():
            val_embeddings = vgae_model.encode(val_data.x, val_data.edge_index)
            auc_val, ap_val = vgae_model.test(val_embeddings, val_data.pos_edge_label_index, val_data.neg_edge_label_index)        

        val_auc_scores.append(auc_val)
        val_ap_scores.append(ap_val)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Training Loss {train_loss:.4f}, Validaion AUC {auc_val:.4f}, Validation AP {ap_val:.4f}")    
    
    generate_plots_for_link_prediction(val_auc_scores, val_ap_scores, epochs, iteration_num)