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

def augment_dataset(train_ds):
    """
    Augments the training dataset with flipped versions of images and annotations.
    Produces 4x the original data (original + h-flip + v-flip + both flips).
    """
    images, labels = train_ds.tensors

    all_images = [images]
    all_labels = [labels]

    all_images.append(flip(images, [3]))
    all_labels.append(flip(labels, [2]))

    all_images.append(flip(images, [2]))
    all_labels.append(flip(labels, [1]))

    all_images.append(flip(images, [2, 3]))
    all_labels.append(flip(labels, [1, 2]))

    return TensorDataset(cat(all_images, dim=0), cat(all_labels, dim=0))

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
        from train_mod import train

        class_counts, rgb_mean = distrib(train_dl)
        total_pixels = class_counts.sum().float()
        num_classes = len(class_counts)
        class_weights = total_pixels / (num_classes * class_counts.float() + 1e-6)
        class_weights = class_weights / class_weights.mean()
        class_weights = class_weights.clamp(min=0.1, max=10.0)

        train_dl = augment_dataset(train_dl)

        netspec_opts = {
            'num_classes': 36,
        }

        train_opts = {
            'lr': 0.001,
            'num_epochs': 120,
            'momentum': 0.9,
            'batch_size': 24,
            'step_size': 50,
            'gamma': 0.1,
            'weight_decay': 1e-4,
            'objective': CrossEntropyLoss(weight=class_weights, label_smoothing=0.1),
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'patience': 30,
        }

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