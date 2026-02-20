from torch import load, save, cat, no_grad
from torch.nn import Softmax
from os.path import exists
from zipfile import ZipFile
from sys import argv
from cnn_categorization_base import cnn_categorization_base
from cnn_categorization_improved import cnn_categorization_improved


def create_submission(model_type):
    data_path = "{}_image_categorization_dataset.pt".format(model_type)
    model_path = "{}-model.pt".format(model_type)

    assert exists(data_path), f"The data file {data_path} does not exist"
    assert exists(model_path), f"The trained model {model_path} does not exist"

    dataset = load(data_path, weights_only=False)
    data_te = dataset["data_te"]
    sets_tr = dataset["sets_tr"]
    data_val = dataset["data_tr"][sets_tr == 2]

    model_state = load(model_path, weights_only=False)
    if model_type == 'base':
        model = cnn_categorization_base(model_state['specs'])
    else:
        model = cnn_categorization_improved(model_state['specs'])

    model.load_state_dict(load(model_state['state'], weights_only=False))
    model.eval()
    soft_max = Softmax(dim=1)

    batch_size = 128
    with no_grad():
        probs_te = []
        for i in range(0, len(data_te), batch_size):
            probs_te.append(soft_max(model(data_te[i:i+batch_size]).squeeze()))
        prob_test = cat(probs_te, dim=0)

        probs_val = []
        for i in range(0, len(data_val), batch_size):
            probs_val.append(soft_max(model(data_val[i:i+batch_size]).squeeze()))
        prob_val = cat(probs_val, dim=0)

    assert prob_test.size() == (9600, 16)
    assert prob_val.size() == (6400, 16)

    output_name_zip = "./{}_categorization.zip".format(model_type)
    output_name_test = "./{}_testing.pt".format(model_type)
    output_name_val = "./{}_validation.pt".format(model_type)
    save(prob_test, output_name_test)
    save(prob_val, output_name_val)
    with ZipFile(output_name_zip, 'w') as zipf:
        zipf.write(model_path)
        zipf.write(output_name_test)
        zipf.write(output_name_val)
        if model_type == "improved":
            if not exists("submission_details.txt"):
                raise FileNotFoundError("Please create submission_details.txt")
            zipf.write("submission_details.txt")


if __name__ == '__main__':
    m_type = "base"
    try:
        m_type = argv[1]
    except IndexError:
        pass
    create_submission(m_type)
