# Implement two surrogate metrics:
# 1) How close are the weight vectors (Cosine similarity or Euclidean distance - need to normalise weight vector for eculidean before?)
# 2) How close are the activations under these models? (e.g. correlation as in https://arxiv.org/pdf/2305.14585.pdf, which ignores the scale - good)
# 3) How close are the decisions by these models? (e.g. ratio of mistakes)

# Problems with 2 + 3: Can have the same activations/same decisions for completely different reasons/weight vectors, especially, when neurons are highly correlated
# Problem with 1: Can not differentiate between scale of neurons (e.g. if one neuron is ten times larger than the other, we expect the weight entry to be only the tenth for their product to be on the same scale)

# Hopefully, if both things are close, our surrogate model should be good

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import matthews_corrcoef
import numpy as np
import os
import torch
from utils.data import load_datasets_reduced
from utils.models import load_model, load_cifar_model
import argparse
import yaml
import logging
import csv

from explain import load_explainer

def load_surrogate(model_name, model_path, device,
                     class_groups, dataset_name, dataset_type,
                     data_root, batch_size, save_dir,
                     validation_size,
                     # num_batches_per_file, start_file, num_files,
                     xai_method,
                     #accuracy,
                     num_classes, C_margin, imagenet_class_ids,testsplit
                     ):
    # (explainer_class, kwargs)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not torch.cuda.is_available():
        device="cpu"
    ds_kwargs = {
        'data_root': data_root,
        'class_groups': class_groups,
        'image_set': "test",
        'validation_size': validation_size,
        "only_train": False,
        'imagenet_class_ids':imagenet_class_ids,
        'testsplit': testsplit
    }

    train, test = load_datasets_reduced(dataset_name, dataset_type, ds_kwargs)
    if dataset_name=="CIFAR":
        model=load_cifar_model(model_path,dataset_type,num_classes,device)
    else:
        model = load_model(model_name, dataset_name, num_classes).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    explainer_cls, kwargs=load_explainer(xai_method, model_path, save_dir, dataset_name)
    explainer = explainer_cls(model=model, dataset=train, device=device, **kwargs)
    explainer.train()
    if C_margin is not None:
        kwargs["C"]=C_margin
    print(f"Checking surrogate faithfulness of {explainer_cls.name}")

    model_weights = model.classifier.weight.detach() #and bias?
    surrogate_weights = explainer.learned_weight.detach() # check that correct dimension
    print("Cosine similarity of weight matrices:", surrogate_faithfulness_cosine(model_weights, surrogate_weights))
    loader=torch.utils.data.DataLoader(train, len(train), shuffle=False) #concat train and test and check activations on both
    x, y = next(iter(loader)) #tqdm.tqdm(loader)
    model_logits = model(x).detach()
    model_predictions = torch.argmax(model_logits, dim=1)

    model_preactivations = explainer.samples
    surrogate_logits = torch.matmul(model_preactivations, surrogate_weights.T)
    surrogate_predictions = torch.argmax(surrogate_logits, dim=1)

    print("Correlation of logits:", surrogate_faithfulness_logits(model_logits, surrogate_logits))
    print("Correlation of prediction:", surrogate_faithfulness_prediction(model_predictions, surrogate_predictions))

    results_dict = [{"Metric": "Cosine similarity of weight matrices", "Score": surrogate_faithfulness_cosine(model_weights, surrogate_weights)},
                    {"Metric": "Correlation of logits", "Score": surrogate_faithfulness_logits(model_logits, surrogate_logits)},
                    {"Metric": "Correlation of prediction", "Score": surrogate_faithfulness_prediction(model_predictions, surrogate_predictions)}]
    with open(os.path.join(save_dir ,"results.csv"), "w") as file: 
        writer = csv.DictWriter(file, fieldnames = ['Metric', 'Score'])
        writer.writeheader()
        writer.writerows(results_dict)


def surrogate_faithfulness_cosine(model_weights, surrogate_weights):
    score = np.average(np.diag(cosine_similarity(model_weights.numpy(), surrogate_weights.numpy())))
    return score

def surrogate_faithfulness_logits(model_logits, surrogate_logits):
    score = np.average(np.diag(cosine_similarity(model_logits.numpy(), surrogate_logits.numpy())))
    return score
    
def surrogate_faithfulness_prediction(model_predictions, surrogate_predictions):
    score = matthews_corrcoef(model_predictions.numpy(), surrogate_predictions.numpy())
    return score

# talk to Galip how to get weight vector and bias vector from surrogate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, "r") as stream:
        try:
            train_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)

    save_dir = f"{train_config['save_dir']}/{os.path.basename(config_file)[:-5]}"

    load_surrogate(model_name=train_config.get('model_name', None),
                     model_path=train_config.get('model_path', None),
                     device=train_config.get('device', 'cuda'),
                     class_groups=train_config.get('class_groups', None),
                     dataset_name=train_config.get('dataset_name', None),
                     dataset_type=train_config.get('dataset_type', 'std'),
                     data_root=train_config.get('data_root', None),
                     batch_size=train_config.get('batch_size', None),
                     save_dir=train_config.get('save_dir', None),
                     validation_size=train_config.get('validation_size', 2000),
                     #accuracy=train_config.get('accuracy', False),
                     #num_batches_per_file=train_config.get('num_batches_per_file', 10),
                     #start_file=train_config.get('start_file', 0),
                     #num_files=train_config.get('num_files', 100),
                     xai_method=train_config.get('xai_method', None),
                     num_classes=train_config.get('num_classes'),
                     C_margin=train_config.get('C',None),
                     imagenet_class_ids=train_config.get('imagenet_class_ids',[i for i in range(397)]),
                     testsplit=train_config.get('testsplit',"test")
                     )
