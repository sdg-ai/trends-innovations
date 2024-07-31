
---

# Trends & Innovations Classifier

## Directory Structure
- **/(archived) initial_classifier**: Contains previous approaches to the classification problem.
- **/app**: Contains the initial deployment of the model within a small webserver.
- **/datasets**: Contains all relevant data used for training, both human-annotated and with generated labels.
- **/experiments**: Contains scripts and notebooks detailing initial Bert-based classification approaches, article generation, and the labeling procedure.
- **/training**: Contains all relevant code used to train the current Bert-based models.
- **/testing_app**: Contains a small streamlit app that can be used to directly test the model after training. To do so simply copy one of the entire checkpoints folder from the training/results directory to the testing_app/checkpoint directory, update the path in the script, and run the streamlit app using the command `streamlit run app.py` in the testing_app directory.

## How to Train a Model
Before you start training, ensure you have the correct Conda environment set up by running the following command in the training directory:
```
conda env create -f environment.yml
```
1. **/training/train**: Contains the main script for the training procedure and demonstrates how the training session is configured.
2. **/training/train_configs.yml**: Contains configurations for the training runs. Each configuration corresponds to one run. These configurations can be used to modify any parameter in the training setup. The available parameters are listed in the top part of `train.py`.
3. **/training/train.sh**: A shell script that can be used to run the training within a `tmux` session on a platform like Azure.
4. when trying to run the training script you will be asked to provide your W&B credentials. Simply follow the instructions or check out the [guide](https://docs.wandb.ai/quickstart).
5. finally after  a successfull run the **training/results** directory will contain all the details of the previous run, including a json of the config and the final predictions as a .csv file.
--- 