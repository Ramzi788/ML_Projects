from semantic_segmentation_base import SemanticSegmentationBase
from semantic_segmentation_improved import SemanticSegmentationImproved
from create_dataset import create_dataset
from train import train
from utils import distrib
from torch import save, random
from torch.nn import CrossEntropyLoss
from argparse import ArgumentParser
import matplotlib.pyplot as plt

# seeding the random number generator. You can disable the seeding for the improvement model
random.manual_seed(0)


def semantic_segmentation(model_type="base"):
    """
    sets up and trains a semantic segmentation model

    Arguments
    ---------
    model_type:  (String) a string in {'base', 'improved'} specifying the targeted model type
    """
    
    # the dataset
    train_dl, val_dl = create_dataset("semantic_segmentation_dataset.pt")

    # an optional export directory
    exp_dir = f"{model_type}_models"

    if model_type == "base":
        # specify netspec_opts
        netspec_opts = {
            'kernel_size': [3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 1, 4, 1, 0, 4],
            'num_filters': [16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128, 36, 36, 36, 36, 36],
            'stride':      [1, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 1, 4, 1, 0, 2],
            'layer_type':  ['conv', 'bn', 'relu', 'conv', 'bn', 'relu', 'conv', 'bn', 'relu',
                            'conv', 'bn', 'relu', 'conv', 'convt', 'skip', 'sum', 'convt'],
            'input':       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 6, (15, 14), 16]
        }

        # specify train_opts
        train_opts = {
            'lr': 0.1,
            'num_epochs': 34,
            'momentum': 0.9,
            'batch_size': 24,
            'step_size': 30,
            'gamma': 0.1,
            'weight_decay': 0.001,
            'objective': CrossEntropyLoss()
        }

        model = SemanticSegmentationBase(netspec_opts)

    elif model_type == "improved":

        # specify netspec_opts

        # specify train_opts

        model = SemanticSegmentationImproved(netspec_opts)
    else:
        raise ValueError(f"Error: unknown model type {model_type}")

    # train the model
    train(model, train_dl, val_dl, train_opts, exp_dir=exp_dir)

    # save model's state and architecture to the base directory
    model = {"state": model.state_dict(), "specs": netspec_opts}
    save(model, f"{model_type}_semantic-model.pt")

    plt.savefig(f"{model_type}_semantic.png")
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_type", default="base", type=str, help="Specify model type")
    args, _ = parser.parse_known_args()

    semantic_segmentation(**args.__dict__)