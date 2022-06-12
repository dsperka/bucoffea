import os
import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from bucoffea.helpers.paths import bucoffea_path

def load_pytorch_model(model_dir: str):
    """
    Given the path to the model directory where the PyTorch model has been 
    saved as a model.pt file, load the model into the script.
    """
    model_file = bucoffea_path(os.path.join(model_dir, "model.pt"))
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
