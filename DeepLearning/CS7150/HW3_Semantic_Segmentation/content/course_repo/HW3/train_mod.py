from torch import optim, save, unique
from torch.utils.data import DataLoader
from os import path, mkdir
import matplotlib.pyplot as plt
import copy
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_ds, val_ds, train_opts, exp_dir=None):
    """
     Fits a semantic segmentation model on the provided data

    Arguments
    ---------
    model: (nn.Module), the segmentation model to train
    train_ds: (TensorDataset), the examples (images and annotations) in the training set
    val_ds: (TensorDataset), the examples (images and annotations) in the validation set
    train_opts: (dict), the training schedule. Read the assignment handout
                for details on the keys and values expected in train_opts
    exp_dir: (string), a directory where the model checkpoints will be saved (optional)

    """

    model.to(device)

    train_dl = DataLoader(train_ds, train_opts["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, len(val_ds), shuffle=False)

    num_tr = train_ds.tensors[0].size(0)
    num_val = val_ds.tensors[0].size(0)

    print(f"Training on {num_tr} and validating on {num_val} images (device: {device})")

    # We use the Adam optimizer for faster and smoother convergence
    # compared to SGD used in the base model
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_opts["lr"],
        weight_decay=train_opts["weight_decay"]
    )

    # We use cosine annealing to smoothly decay the learning rate
    # from the initial value down to near zero over training
    num_epochs = train_opts["num_epochs"]
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    # the loss function (weighted cross entropy with label smoothing)
    criterion = train_opts["objective"].to(device)

    # track the training metrics
    tr_loss = []
    val_loss = []
    pixel_acc = []
    per_class_acc = []
    iu_score = []

    # track the best model by per-class accuracy and use early stopping
    # to prevent overfitting when validation accuracy stops improving
    best_class_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = train_opts.get("patience", 25)
    no_improve = 0

    for epoch in range(num_epochs):

        # training phase
        model.train()
        e_loss_tr, _ = fit(model, train_dl, criterion, optimizer)
        tr_loss.append(e_loss_tr)

        # validation phase
        model.eval()
        e_loss_val, predictions = fit(model, val_dl, criterion)

        lr_scheduler.step()
        e_per_class_acc, e_pixel_acc, e_iu_score = accuracy_metrics(predictions.cpu(), val_ds.tensors[1])

        val_loss.append(e_loss_val)
        pixel_acc.append(e_pixel_acc)
        per_class_acc.append(e_per_class_acc)
        iu_score.append(e_iu_score)

        # it is always good to report the training metrics at the end of every epoch
        lr_now = optimizer.param_groups[0]['lr']
        print(f"[{epoch + 1}/{num_epochs}: tr_loss {e_loss_tr:.5} val_loss {e_loss_val:.5} "
              f"class_acc {e_per_class_acc:.2%} pixel_acc {e_pixel_acc:.2%} iu_score {e_iu_score:.2%} lr {lr_now:.6f}]")

        # save the best model weights when per-class accuracy improves
        if e_per_class_acc > best_class_acc:
            best_class_acc = e_per_class_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        # stop training early if no improvement for too many epochs
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best class_acc: {best_class_acc:.2%}")
            break

        # save model checkpoint if exp_dir is specified
        if exp_dir:
            if path.exists(exp_dir):
                save(model.state_dict(), path.join(exp_dir, f"checkpoint_{epoch + 1}.pt"))
            else:
                try:
                    mkdir(exp_dir)
                    save(model.state_dict(), path.join(exp_dir, f"checkpoint_{epoch + 1}.pt"))
                except FileNotFoundError:
                    pass

    # restore the best model weights found during training
    print(f"Restoring best model with class_acc: {best_class_acc:.2%}")
    model.load_state_dict(best_model_wts)

    # move model back to CPU for saving and submission
    model.cpu()

    # plot the training metrics at the end of training
    plot(tr_loss, val_loss, per_class_acc, pixel_acc, iu_score)


def fit(model, data_dl, criterion, optimizer=None):
    """
    Executes a training (or validation) epoch

    Arguments
    --------
    model: (nn.Module), the segmentation model
    data_dl: (DataLoader), the dataloader of the training or validation set
    criterion: The objective function
    optimizer: The optimization function (optional)

    Returns
    ------
    e_loss: (float), the average loss on the given set for the epoch
    predictions: (Tensor), the pixel-level predictions for the epoch. Computed only on the validation set for the accuracy
                metrics
    """
    e_loss = 0

    predictions = []
    for images, annots in data_dl:
        images = images.to(device)
        annots = annots.to(device)
        pred = model(images)
        loss = criterion(pred, annots)
        e_loss += loss.item()

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            predictions = pred.argmax(dim=1)

    e_loss = e_loss/len(data_dl)
    return e_loss, predictions


def accuracy_metrics(predictions, labels):

    """
    Computes the three accuracy metrics as explained in the assignment handout

    Arguments
    --------
    predictions: (Tensor), the pixel-level predictions for a mini-batch of images
    labels: (Tensor), the corresponding ground-truth labels of the mini-batch
    Returns
    -------
    per_class_acc: (float), the per-class accuracy as described in the handout
    pixel_acc: (float), the pixel accuracy as described in the handout
    iu_score: (float), the intersection over union score as described in the handout

    """
    per_class_acc = 0
    iu_score = 0
    pixel_acc = predictions.eq(labels).float().mean().item()

    classes = unique(labels)

    # compute the per-class and IoU accuracies for each class
    for class_i in classes:
        predictions_i = predictions.eq(class_i)
        labels_i = labels.eq(class_i)

        correct_predictions = (predictions_i & labels_i).sum().item()
        num_class_i = labels_i.sum().item()

        per_class_acc += correct_predictions / num_class_i
        iu_score += correct_predictions / (predictions_i | labels_i).sum().item()

    # average the accuracies over all the classes
    per_class_acc = per_class_acc / len(classes)
    iu_score = iu_score / len(classes)

    return per_class_acc, pixel_acc, iu_score


def plot(loss_tr, loss_val, per_class_acc, pixel_acc, iu_score):
    """
    plots the training metrics

    Arguments
    ---------
    loss_tr: (list), the average epoch loss on the training set for each epoch
    loss_val: (list), the average epoch loss on the validation set for each epoch
    per_class_acc: (list), the average epoch per-class accuracy for each epoch
    pixel_acc: (list), the average epoch pixel accuracy for each epoch
    iu_score: (list), the average epoch IoU score for each epoch

    """
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    n = [i + 1 for i in range(len(loss_tr))]

    ax1.plot(n, loss_tr, 'bs-', markersize=3, label="train")
    ax1.plot(n, loss_val, 'rs-', markersize=3, label="validation")
    ax1.legend()
    ax1.set_title("Training and Validation Losses")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.plot(n, per_class_acc, 'go-', markersize=3, label="Class Accuracy")
    ax2.plot(n, pixel_acc, 'bo-', markersize=3, label="Pixel Accuracy")
    ax2.plot(n, iu_score, 'ro-', markersize=3, label="IoU Score")
    ax2.legend()
    ax2.set_title("Validation Accuracy Metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")