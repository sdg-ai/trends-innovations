import argparse
import yaml
import os
from app.models.transformer import transformers_lib, TransformerTandIClassifier
from app.data import get_data_loaders

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
args = parser.parse_args()

if __name__ == "__main__":
    # load train_config yaml file
    with open("./train_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model_config = config["transformers"]
    model_config["num_labels"] = config["num_labels"]
    model_config["dataset_splits"] = config["dataset_splits"]
    # check if model_name argument is transformer or not
    if args.model_name in transformers_lib.keys():
        model = TransformerTandIClassifier(
            model_name=args.model_name,
            model_config=model_config,
            save_model_dir=config["save_model_dir"],
            wandb_config=config["wandb_config"])
        data_loaders = get_data_loaders(config["data_dir"], model, config["transformers"]["batch_sizes"])
        model.train(data_loaders["train"], data_loaders["val"])
        model.test(data_loaders["test"])
    elif args.model_name == "svc":
        model = None
    elif args.model_name == "mnb":
        model = None
    else:
        raise ValueError(
            f"model_name argument must be one of the following: svc, mnb, or one of {list(transformers_lib.keys())}")
