from semantic_segmentation_base import SemanticSegmentationBase
from semantic_segmentation_improved import SemanticSegmentationImproved
from create_dataset import create_dataset
from train import train as train_base
from train_mod import train as train_improved
from utils import distrib
from torch import save, random, cat, flip, load
from torch.utils.data import TensorDataset, Dataset
from torch.nn import CrossEntropyLoss
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch

# seeding the random number generator. You can disable the seeding for the improvement model
random.manual_seed(0)


class AugmentedSegDataset(Dataset):
    """
    Custom dataset that applies on-the-fly augmentation using torchvision.transforms.
    """
    def __init__(self, images, annotations):
        self.images = images
        self.annotations = annotations.long()
        self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
        self.tensors = (images, annotations)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].clone()
        anno = self.annotations[idx].clone()

        if torch.rand(1).item() > 0.5:
            image = torch.flip(image, [2])
            anno = torch.flip(anno, [1])

        if torch.rand(1).item() > 0.5:
            image = torch.flip(image, [1])
            anno = torch.flip(anno, [0])

        image = self.color_jitter(image)

        return image, anno


def augment_dataset(train_ds):
    """
    Pre-computes 4x augmented data with flips.
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

        # train the model
        train_base(model, train_dl, val_dl, train_opts, exp_dir=exp_dir)

    elif model_type == "improved":
        # use ALL data (train + val) for training to maximize data
        data = load("semantic_segmentation_dataset.pt", weights_only=False)
        all_images = data['images_tr']
        all_annots = data['anno_tr']

        # compute class weights for class imbalance
        train_all = TensorDataset(all_images, all_annots)
        class_counts, rgb_mean = distrib(train_all)
        total_pixels = class_counts.sum().float()
        num_classes = len(class_counts)
        class_weights = total_pixels / (num_classes * class_counts.float() + 1e-6)
        class_weights = class_weights / class_weights.mean()
        class_weights = class_weights.clamp(min=0.1, max=10.0)

        # pre-compute 4x flipped data + wrap with on-the-fly ColorJitter
        train_flipped = augment_dataset(train_all)
        flipped_images, flipped_annots = train_flipped.tensors
        train_aug = AugmentedSegDataset(flipped_images, flipped_annots)

        # specify netspec_opts
        netspec_opts = {
            'num_classes': 36,
        }

        # specify train_opts
        train_opts = {
            'lr': 0.001,
            'num_epochs': 150,
            'momentum': 0.9,
            'batch_size': 24,
            'step_size': 50,
            'gamma': 0.1,
            'weight_decay': 1e-4,
            'objective': CrossEntropyLoss(weight=class_weights, label_smoothing=0.05),
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'patience': 40,
        }

        model = SemanticSegmentationImproved(netspec_opts)

        # train the model
        train_improved(model, train_aug, val_dl, train_opts, exp_dir=exp_dir)

    else:
        raise ValueError(f"Error: unknown model type {model_type}")

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