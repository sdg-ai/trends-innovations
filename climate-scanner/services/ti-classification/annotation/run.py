import argparse
import os
import getpass
from dotenv import load_dotenv
from simple_annotation_runner import run_simple_annotation

load_dotenv()

def get_api_credentials():
    # OpenAI configuration
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-4")

    # Azure OpenAI configuration
    azure_openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    azure_openai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")

    if not (openai_api_key or (azure_openai_api_key and azure_openai_endpoint)):
        # Try Azure OpenAI first
        if os.environ.get("AZURE_OPENAI_ENDPOINT"):
            os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure OpenAI: ")
            azure_openai_api_key = os.environ["AZURE_OPENAI_API_KEY"]
        else:
            # Fallback to OpenAI
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
            openai_api_key = os.environ["OPENAI_API_KEY"]

    return {
        "openai_api_key": openai_api_key,
        "openai_model": openai_model,
        "azure_openai_api_key": azure_openai_api_key,
        "azure_openai_endpoint": azure_openai_endpoint,
        "azure_openai_deployment": azure_openai_deployment,
        "azure_openai_api_version": azure_openai_api_version
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Run annotation on articles dataset')
    parser.add_argument('--data-dir', type=str, default='./dataset',
                       help='Directory containing the dataset (default: ./dataset)')
    parser.add_argument('--num-category-groups', type=int, default=1,
                       help='Number of category groups (default: 1)')
    parser.add_argument('--num-majority-vote', type=int, default=1,
                       help='Number of votes required for majority (default: 1)')
    parser.add_argument('--conflict-resolution', type=str, default='majority_vote',
                       choices=['majority_vote', 'unsure'],
                       help='Strategy for resolving conflicts (default: majority_vote)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Enable benchmarking against human annotations. This will not annotate the data, but just evaluate the model.')
    return parser.parse_args()

def main():
    args = parse_args()
    credentials = get_api_credentials()
    
    run_simple_annotation(
        data_dir=args.data_dir,
        num_category_groups=args.num_category_groups,
        num_majority_vote=args.num_majority_vote,
        conflict_resolution_strategy=args.conflict_resolution,
        benchmark=False,
        **credentials
    )

if __name__ == '__main__':
    main()