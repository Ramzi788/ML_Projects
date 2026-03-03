from torch import load, save, argmax, zeros
from torch.nn import Softmax
from os.path import exists
from zipfile import ZipFile
from argparse import ArgumentParser
from semantic_segmentation_base import SemanticSegmentationBase
from semantic_segmentation_improved import SemanticSegmentationImproved


def create_submission(model_type, batch_size):
    """
    Evaluates the model on the test and validation data and creates a submission file

    Arguments
    ---------
    model_type (string): Specifies the model type for which the submission is being made for.

    """
    model_path = f"{model_type}_semantic-model.pt"

    if not exists(model_path):
        raise ValueError("Error: the trained model {} does not exits".format(model_path))

    dataset = load("semantic_segmentation_dataset.pt")

    data_te = dataset["images_te"]
    sets_tr = dataset["sets_tr"]
    data_val = dataset["images_tr"]
    data_val = data_val[sets_tr == 2]

    del dataset

    model_state = load(model_path)
    if model_type == 'base':
        model = SemanticSegmentationBase(model_state['specs'])
    else:
        model = SemanticSegmentationImproved(model_state['specs'])

    model.load_state_dict(model_state['state'])
    model.eval()

    # test set
    pred_test = evaluate(data_te, model, batch_size)
    if pred_test.size() != (400, 32, 32):
        raise ValueError(f"Expected the output of the test set to be size (400, 32, 32)) "
                         f"but got {pred_test.size()} instead")
    # validation set
    pred_val = evaluate(data_val, model, batch_size)
    if pred_val.size() != (200, 32, 32):
        raise ValueError(f"Expected the output of the validation set to be of size (200, 32, 32) "
                         f"but got {pred_val.size()} instead")

    output_name_zip = "./{}_submission.zip".format(model_type)
    output_name_test = "./{}_testing.pt".format(model_type)
    output_name_val = "./{}_validation.pt".format(model_type)

    save(pred_test, output_name_test)
    save(pred_val, output_name_val)

    with ZipFile(output_name_zip, 'w') as zipf:
        zipf.write(model_path)
        zipf.write(output_name_test)
        zipf.write(output_name_val)

        if model_type == "improved":
            if not exists("submission_details.txt"):
                raise FileNotFoundError("Please create a file submission_details.txt describing your improvements")
            else:
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


if __name__ == '__main__':
    # change model_type and batch_size to suit your needs
    parser = ArgumentParser()
    parser.add_argument("--model_type", default="base", type=str, help="Specify model type")
    parser.add_argument("--batch_size", default=100, type=int, help="specify the batch size")
    args, _ = parser.parse_known_args()

    create_submission(**args.__dict__)
