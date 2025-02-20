from typing import Dict, List, Set
import streamlit as st
import os
from model import TIClassifier
from pathlib import Path
import pandas as pd

def get_available_models() -> Dict[str, List[str]]:
    """
    Returns a dictionary mapping model names to a list of seeds for that model.

    Returns:
        Dict[str, List[str]]: A dictionary mapping model names to a list of seeds for that model.
    """
    # Get the path to the checkpoints directory
    checkpoint_dir = Path('./training/results/checkpoints')
    # Dictionary to store models and their seeds
    model_seeds: Dict[str, Set[str]] = {}
    
    if checkpoint_dir.exists():
        for item in checkpoint_dir.iterdir():
            if item.is_dir():
                model_name = item.name
                # Initialize empty set for this model's seeds
                model_seeds[model_name] = set()
                # Check for seed directories
                for seed_dir in item.iterdir():
                    if seed_dir.is_dir() and seed_dir.name.startswith('seed_'):
                        model_seeds[model_name].add(seed_dir.name)
    
    # Convert sets to sorted lists and return
    return {model: sorted(seeds) for model, seeds in model_seeds.items()}

# Initialize the classifier
@st.cache_resource
def init_classifier(checkpoint: str, seed: str) -> TIClassifier:
    """
    Initializes a TIClassifier object with the specified checkpoint and seed.

    Args:
        checkpoint (str): The name of the checkpoint directory.
        seed (str): The seed used during training.

    Returns:
        TIClassifier: A TIClassifier object initialized with the specified checkpoint and seed.
    """
    classifier = TIClassifier(
        checkpoint_dir=f'./training/results/checkpoints/{checkpoint}',
        seed=seed
    )
    return classifier.load_model()

def main():
    st.set_page_config(
        page_title="TandI-Classifier",
        page_icon="",
        layout="wide"
    )
    
    st.title("Trends and Innovations Classifier")
    #st.write("Enter text to classify trends and innovations")
    
    # Get available models and their associated seeds
    models_and_seeds = get_available_models()
    
    if not models_and_seeds:
        st.error("No models found in the checkpoints directory!")
        return
        
    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox(
            "Select Model",
            sorted(models_and_seeds.keys()),
            index=0
        )
    
    with col2:
        # Get seeds for the selected model
        available_seeds = models_and_seeds.get(selected_model, [])
        if not available_seeds:
            st.error(f"No seeds found for model {selected_model}")
            return
            
        selected_seed = st.selectbox(
            "Select Seed",
            available_seeds,
            index=0
        )
    
    # Initialize model
    try:
        classifier = init_classifier(selected_model, selected_seed)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Text input
    user_input = st.text_area("Enter text for classification:", height=200)
    
    if st.button("Classify"):
        if user_input:
            with st.spinner('Analyzing text...'):
                try:
                    results = classifier.predict(user_input)
                    
                    # Display results in a table
                    st.subheader("Classification Results")
                    
                    # Convert results to a format suitable for display
                    results_data = {
                        "Label": [label for label, _ in results],
                        "Confidence": [prob * 100 for _, prob in results]  # Convert to percentage
                    }
                    
                    # Create a styled dataframe
                    df = pd.DataFrame(results_data)
                    
                    # Style the dataframe
                    def highlight_max(df):
                        max_confidence = df['Confidence'].max()
                        # Using a darker, more muted green that works better with dark theme
                        highlight_color = 'rgba(76, 175, 80, 0.3)'  # Semi-transparent green
                        return pd.DataFrame(
                            [['background-color: ' + highlight_color + '; color: white;' if v == max_confidence else '' for v in df['Confidence']],
                             ['background-color: ' + highlight_color + '; color: white;' if v == max_confidence else '' for v in df['Confidence']]],
                            index=['Label', 'Confidence']
                        ).T
                    
                    styled_df = df.style.apply(highlight_max, axis=None)
                    
                    # Add custom CSS to ensure better text visibility
                    st.markdown("""
                        <style>
                        .stDataFrame {
                            font-size: 1.1rem;
                        }
                        .stDataFrame td {
                            font-weight: 500;
                            color: rgba(250, 250, 250, 0.95) !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                    
                    # Display the table with custom formatting
                    st.dataframe(
                        styled_df,
                        column_config={
                            "Label": st.column_config.TextColumn(
                                "Label",
                                help="Classification category"
                            ),
                            "Confidence": st.column_config.NumberColumn(
                                "Confidence",
                                help="Prediction confidence score",
                                format="%.2f%%"  # Format as percentage with 2 decimal places
                            )
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error during classification: {str(e)}")
        else:
            st.warning("Please enter some text to classify!")

if __name__ == "__main__":
    main()