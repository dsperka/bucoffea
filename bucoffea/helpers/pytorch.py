import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

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

def prepare_data_for_dnn(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """
    Scales the data to be ready for the deep neural network to zero mean and unit variance.
    Ignores NaN and infinity values in data, returns the data as a Pandas DataFrame.

    "mask" parameter specifies the subset of events to compute the mean and standard deviation for.

    Note that the input data MUST be passed in as a Pandas DataFrame.
    """
    features = df.replace([np.inf, -np.inf], np.nan)

    # We will compute mean and std only for the events we care about per region, hence the mask
    features_to_compute = features.loc[mask]

    # Compute mean and standard deviation per feature, ignoring NaN values
    mean = np.nanmean(features_to_compute, axis=0)
    std = np.nanstd(features_to_compute, axis=0)

    # Do the feature transformation
    features = (features - mean) / std

    return features


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