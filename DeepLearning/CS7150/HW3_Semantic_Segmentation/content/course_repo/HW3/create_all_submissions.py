from argparse import ArgumentParser
from os.path import exists
from pathlib import Path
from zipfile import ZipFile

import torch
from torch import load, argmax, zeros
from torch.nn import Softmax

from semantic_segmentation_base import SemanticSegmentationBase
from semantic_segmentation_improved import SemanticSegmentationImproved


def create_submission(model_type, batch_size):
    """
    Evaluates the model on the test and validation data and creates a submission file

    Arguments
    ---------
    model_type (string): Specifies the model type for which the submission is being made for.
    """

    with torch.no_grad():
        model_path = f"{model_type}_semantic-model.pt"

        assert exists(model_path), f"Error: the trained model {model_path} does not exist"

        dataset = load("semantic_segmentation_dataset.pt")
        data_val = dataset["images_tr"][dataset["sets_tr"] == 2]
        data_te = dataset["images_te"]

        model_state = load(model_path)
        if model_type == "base":
            model = SemanticSegmentationBase(model_state["specs"])
        else:
            model = SemanticSegmentationImproved(model_state["specs"])

        model.load_state_dict(model_state["state"])
        model.eval()

        # test set
        pred_test = evaluate(data_te, model, batch_size)
        assert pred_test.size() == (400, 32, 32), f"Expected the output of the test set to be of size (400, 32, 32) but was {pred_test.size()} instead"

        # validation set
        pred_val = evaluate(data_val, model, batch_size)
        assert pred_val.size() == (200, 32, 32), f"Expected the output of the validation set to be of size (200, 32, 32) but was {pred_val.size()} instead"

        # Save for Kaggle.
        with Path(f"kaggle_{model_type}_test_submission.csv").open(mode="w") as writer:
            writer.write("Id,Category\n")
            for i in range(pred_test.size(0)):
                for j in range(pred_test.size(1)):
                    for k in range(pred_test.size(2)):
                        writer.write(f"{i}_{j}_{k},{pred_test[i, j, k]}\n")

        with Path(f"kaggle_{model_type}_val_submission.csv").open(mode="w") as writer:
            writer.write("Id,Category\n")
            for i in range(pred_val.size(0)):
                for j in range(pred_val.size(1)):
                    for k in range(pred_val.size(2)):
                        writer.write(f"{i}_{j}_{k},{pred_val[i, j, k]}\n")

        # Canvas submission.
        output_name_zip = f"{model_type}_submission.zip"
        output_name_test = f"{model_type}_testing.pt"
        output_name_val = f"{model_type}_validation.pt"

        save(pred_test, output_name_test)
        save(pred_val, output_name_val)

        with ZipFile(output_name_zip, "w") as zipf:
            zipf.write(model_path)
            zipf.write(output_name_test)
            zipf.write(output_name_val)

            if model_type == "improved":
                assert exists("submission_details.txt"), "Please create a file submission_details.txt describing your improvements"
                zipf.write("submission_details.txt")


def evaluate(data, model, batch_size):
    num_examples = data.size(0)
    img_size = 32

    soft_max = Softmax(dim=1)
    pred_vals = zeros(num_examples, img_size, img_size)

    for i in range(0, num_examples, batch_size):
        pred_vals[i: i + batch_size] = argmax(soft_max(model(data[i: i + batch_size])), dim=1).squeeze()

    del data
    return pred_vals.long()


if __name__ == "__main__":
    # change model_type and batch_size to suit your needs
    parser = ArgumentParser()
    parser.add_argument("--model_type", default="base", type=str, help="Specify model type")
    parser.add_argument("--batch_size", default=100, type=int, help="specify the batch size")
    args, __ = parser.parse_known_args()

    create_submission(**args.__dict__)
