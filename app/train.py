import argparse
import yaml
import wandb
import torch
from datetime import datetime
from models.transformer import transformers_lib, TransformerTandIClassifier
from data import get_data_loaders

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
args = parser.parse_args()

if __name__ == "__main__":
    # load train_config yaml file
    current_time = datetime.strftime(datetime.now(), format="%Y.%m.%d-%H:%M:%S")
    with open("app/train_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config["transformer_config"]["model_name"] in transformers_lib.keys():
        for seed in range(config["meta"]["num_seeds"]):
            config["transformer_config"]["seed"] = config["meta"]["initial_seed"] + seed
            config["transformer_config"]["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
            # init weights and biases
            wandb.init(
                entity=config["wandb_config"]["entity"],
                project=config["wandb_config"]["project"],
                config=config["transformer_config"],
                mode="disabled" if config["wandb_config"]["disabled"] else "online",
                group=f"{config['transformer_config']['model_name']}-{current_time}",
                job_type="train",
                name="seed_"+str(config["transformer_config"]["seed"]))
            model = TransformerTandIClassifier(
                model_name=args.model_name,
                model_config=config["transformer_config"],
                save_model_dir=config["meta"]["save_model_dir"]
            )
            data_loaders = get_data_loaders(
                model,
                config["meta"]["data_dir"],
                model.config["batch_sizes"],
                config["meta"]["dataset_splits"]
            )
            model.train(data_loaders["train"], data_loaders["val"],)
            model.test(data_loaders["test"])
    else:
        raise ValueError(f"model_name argument must be one of the following: {list(transformers_lib.keys())}")
