# module that contains GNN models 
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATv2Conv 

# baseline GCN (Graph Convolutional Neural Network)
class GCNBaseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNBaseline, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, return_embeddings=False):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)        
        if (return_embeddings):
            return x

        return torch.log_softmax(x, dim=1)


# baseline GraphSAGE (GraphSAGE Neural Network)
class GraphSAGEBaseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEBaseline, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, return_embeddings=False):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        if (return_embeddings):
            return x
        
        return torch.log_softmax(x, dim=1)

# baseline GAT (Graph Attention Network)
class GATBaseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_of_heads=1):
        super(GATBaseline, self).__init__()
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=num_of_heads)
        self.gat2 = GATv2Conv(hidden_channels*num_of_heads, out_channels, heads=1)    

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x) # exponential linear unit
        x = self.gat2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
    

# custom GCN (Graph Convolutional Neural Network with customizable parameters)
class GCNCustom(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_of_layers, dropout_ratio):
        super(GCNCustom, self).__init__()
        self.layers = torch.nn.ModuleList([GCNConv(in_channels, hidden_channels)])
        for layer_num in range(1, num_of_layers-1):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        
        self.layers.append(GCNConv(hidden_channels, out_channels))
        self.dropout_ratio = dropout_ratio

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        
        x = self.layers[-1](x, edge_index)
        
        return F.log_softmax(x, dim=1)
    
        
# custom GraphSAGE (GraphSAGE Neural Network with customizable parameters)
class GraphSAGECustom(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_of_layers, dropout_ratio, aggregation_type="mean"):
        super(GraphSAGECustom, self).__init__()
        self.layers = torch.nn.ModuleList([SAGEConv(in_channels, hidden_channels, aggr=aggregation_type)])
        for layer_num in range(1, num_of_layers-1):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregation_type))
        
        self.layers.append(SAGEConv(hidden_channels, out_channels, aggr=aggregation_type))
        self.dropout_ratio = dropout_ratio

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        
        x = self.layers[-1](x, edge_index)
        
        return F.log_softmax(x, dim=1)

# custom GAT (Graph Attention Neural Network with customizable parameters)
class GATCustom(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_of_layers, dropout_ratio, num_of_heads=1):
        super(GATCustom, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dropout_ratio = dropout_ratio
        
        self.layers.append(GATv2Conv(in_channels, hidden_channels, num_of_heads, concat=True))
        
        for layer in range(num_of_layers-1):
            self.layers.append(GATv2Conv(hidden_channels*num_of_heads, hidden_channels, num_of_heads, concat=True))        
        
        self.layers.append(GATv2Conv(hidden_channels*num_of_heads, out_channels, heads=num_of_heads, concat=False))
        
    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
            x = layer(x, edge_index)
            x = F.elu(x)

        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.layers[-1](x, edge_index)
        
        return F.log_softmax(x, dim=1)

# Encoder used to approximate normal dist. from which
# node embeddings will be sampled to approximate adjacency 
# matrix which will be used by VGAE (Variational Graph Auto Encoder)      
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2*out_channels)
        self.conv_mu = GCNConv(2*out_channels, out_channels)
        self.conv_logstd = GCNConv(2*out_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
        
        
        