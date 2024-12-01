from dataclasses import dataclass
import torch
@dataclass
class WandbConfig:
    def __init__(self, disable_wandb=False):
        self.disabled = disable_wandb
        self.job_type_modifier = ""
        self.group_name_modifier = ""
        self.project = "Trends and Innovations Classifier | AI For Good"

@dataclass
class RunConfig:
    def __init__(self, debug=False):
        # data details
        self.data_dir = "datasets"
        self.keep_ignores = False
        self.keep_rejects = False
        self.label_reject_and_ignore_as_irrelevant = False
        self.drop_conflicting_answers = True
        self.use_chatgpt_annotated_data = True
        self.use_human_annotated_data = True
        self.combine_categories = False
        self.min_samples_per_label = 10
        self.stratify = True
        self.train_size = 0.7
        self.val_size = 0.2
        self.test_size = 0.1
        self.undersample = False
        self.upsample = False
        self.only_top_n_categories_by = None
        # model details
        self.model_name = "distilbert-base-uncased"
        self.lr = 5e-5
        self.epochs = 30 if not debug else 1
        self.patience = 5
        self.num_warmup_steps = 500
        self.train_batch_size = 32 if not debug else 3
        self.val_batch_size = 64 if not debug else 3
        self.test_batch_size = 64 if not debug else 3
        # other details
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.initial_seed = 1
        self.num_seeds = 3
        self.checkpoints_dir = "results/checkpoints"
        self.skip = False
        self.sweep_config = None
        self.debug = False if not debug else True
