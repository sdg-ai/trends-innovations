import os
import argparse
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute, Environment, BuildContext
from dotenv import load_dotenv

load_dotenv()

AZURE_SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID") or ""
AZURE_RESOURCEGROUP_NAME = os.environ.get("AZURE_RESOURCEGROUP_NAME") or ""
AZURE_ML_WORKSPACE_NAME = os.environ.get("AZURE_ML_WORKSPACE_NAME") or ""

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()

def connect_to_workspace():
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=AZURE_SUBSCRIPTION_ID,
        resource_group_name=AZURE_RESOURCEGROUP_NAME,
        workspace_name=AZURE_ML_WORKSPACE_NAME
    )
    return ml_client


def create_env(ml_client):
    # Name assigned to the compute cluster
    custom_env_name = "aml-tandic-pytorch"
    custom_job_env = Environment(
        name=custom_env_name,
        description="custom env to train trends and innovation classifier models",
        conda_file=os.path.join(os.getcwd(), "environment.yml"),
        image="mcr.microsoft.com/azureml/curated/acpt-pytorch-1.13-cuda11.7:latest"
    )
    custom_job_env = ml_client.environments.create_or_update(custom_job_env)
    print(
        f"Environment with name {custom_job_env.name} is registered to workspace, the environment version is {custom_job_env.version}")
    return custom_job_env


def create_compute_resource(ml_client):
    gpu_compute_target = "tandic-v100"
    cpu_compute_target = "tancic-test-compute"
    compute_target = gpu_compute_target if args.device == "gpu" else cpu_compute_target
    try:
        compute_resource = ml_client.compute.get(compute_target)
        print(f"You already have a cluster named {compute_target}, we'll reuse it as is.")
    except Exception:
        print("Creating a new cpu compute target...")
        compute_resource = AmlCompute(
            name=compute_target,
            size="Standard_NC6s_v3" if args.device == "gpu" else "Standard_DS3_v2",
            # Minimum running nodes when there is no job running
            min_instances=0,
            # Nodes in cluster
            max_instances=1,
            # How many seconds will the node running after the job termination
            idle_time_before_scale_down=180,
            # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
            tier="Dedicated"
        )
        print(
            f"AMLCompute with name {compute_resource.name} will be created, with compute size {compute_resource.size}")
        compute_resource = ml_client.compute.begin_create_or_update(compute_resource)
    return compute_resource


if __name__ == '__main__':
    ml_client = connect_to_workspace()
    compute_resource = create_compute_resource(ml_client)
    env = create_env(ml_client)
    job = command(code="./",  # location of source code
        command="python train.py --model_name distilbert-base-uncased", environment="aml-tandic-pytorch@latest",
        display_name="tandic-distilbert-base-uncased", )
    ml_client.create_or_update(job)
