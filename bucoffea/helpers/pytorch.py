import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler

from bucoffea.helpers.paths import bucoffea_path

def load_pytorch_state_dict(model_dir: str):
    """
    Given the path to the model directory where the state dict of a pre-trained 
    PyTorch model has been saved as a model_state_dict.pt file,
    load the state dict into the script.
    """
    model_file = bucoffea_path(os.path.join(model_dir, "model_state_dict.pt"))
    assert os.path.exists(model_file), f"PyTorch model file does not exist: {model_file}"

    return torch.load(model_file)

def prepare_data_for_dnn(df: pd.DataFrame) -> np.ndarray:
    """
    Scales the data to be ready for the deep neural network (using sklearn's StandardScaler).
    Returns the data as a Numpy array.

    Note that the input data MUST be passed in as a Pandas DataFrame.
    """
    scaler = StandardScaler()

    # Ignore NaN or infinity values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    features = df.to_numpy()

    return scaler.fit_transform(features)

def get_dnn_predictions(model, features: np.ndarray) -> np.ndarray:
    """
    Given the PyTorch DNN model and the set of features, perform a feed-forward
    and obtain the predictions.

    Returns the scores as a NumPy array.
    """
    # Run on a CPU
    device = torch.device("cpu")

    # Transform the data into a Torch tensor before feeding into the model
    x = torch.Tensor(features).to(device)
    # Put model in evaluation mode
    model.eval()

    return model(x).cpu().detach().numpy()


def swish(x):
    return x * torch.sigmoid(x)


class FullyConnectedNN(nn.Module):
    """
    PyTorch based fully connected neural network class that runs on CPU.
    """

    def __init__(self, n_features, n_classes, n_nodes, dropout=0.5):
        super(FullyConnectedNN, self).__init__()
        self.layers = []
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.dropout = dropout
        self._build_layers()

    def _build_layers(self):
        self.layers.append(nn.Linear(self.n_features, self.n_nodes[0]))
        last_nodes = self.n_nodes[0]
        for i_n_nodes in self.n_nodes[1:]:
            self.layers.append(nn.BatchNorm1d(last_nodes))
            self.layers.append(nn.Linear(last_nodes, i_n_nodes))
            last_nodes = i_n_nodes
            self.layers.append(nn.Dropout(self.dropout))
        self.layers.append(nn.Linear(last_nodes, self.n_classes))
        self.layers.append(nn.BatchNorm1d(self.n_classes))

        for i, layer in enumerate(self.layers):
            setattr(self, "layer_%d" % i, layer)

    def forward(self, x):
        """Forward pass through the network."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = swish(layer(x))
            else:
                x = layer(x)
        x = F.softmax(x, dim=1)
        return x

    def predict(self, x):
        """Make predictions on input data."""
        x = torch.Tensor(x).to(torch.device("cpu"))
        # Put the model into evaluation mode i.e. self.train(False)
        self.eval()
        return self(x).cpu().detach().numpy()