from torch.utils.data import TensorDataset
from torch import load


def create_dataset(data_path):
    """
    Reads the data and prepares the training and validation sets. No preprocessing is required.

    Arguments
    ---------
    data_path: (string),  the path to the file containing the data

    Return
    ------
    train_ds: (TensorDataset), the examples (inputs and labels) in the training set
    val_ds: (TensorDataset), the examples (inputs and labels) in the validation set
    """
    data = load(data_path)

    images_tr = data['images_tr']
    anno_tr = data['anno_tr']
    sets_tr = data['sets_tr']

    train_mask = (sets_tr == 1)
    val_mask = (sets_tr == 2)

    train_ds = TensorDataset(images_tr[train_mask], anno_tr[train_mask])
    val_ds = TensorDataset(images_tr[val_mask], anno_tr[val_mask])

    return train_ds, val_ds