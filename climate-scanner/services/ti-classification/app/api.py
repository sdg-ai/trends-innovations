from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from model import TIClassifier
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="Trends and Innovations Classifier API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

class ModelConfig(BaseModel):
    model_name: str
    seed: str

# Global variables to store current model configuration
current_model: Optional[str] = None
current_seed: Optional[str] = None
classifier: Optional[TIClassifier] = None

def get_available_models() -> Dict[str, List[str]]:
    """Returns a dictionary mapping model names to their available seeds"""
    checkpoint_dir = Path('./training/results/checkpoints')
    model_seeds = {}
    
    if checkpoint_dir.exists():
        for item in checkpoint_dir.iterdir():
            if item.is_dir():
                model_name = item.name
                seeds = []
                for seed_dir in item.iterdir():
                    if seed_dir.is_dir() and seed_dir.name.startswith('seed_'):
                        seeds.append(seed_dir.name)
                if seeds:
                    model_seeds[model_name] = sorted(seeds)
    
    return model_seeds

def load_model(model_name: str, seed: str) -> TIClassifier:
    """Load a model with the specified configuration"""
    classifier = TIClassifier(
        checkpoint_dir=f'./training/results/checkpoints/{model_name}',
        seed=seed
    )
    return classifier.load_model()

@app.get("/models")
async def list_models():
    """Get available models and their seeds"""
    models = get_available_models()
    if not models:
        raise HTTPException(status_code=404, detail="No models found in the checkpoints directory")
    
    return {
        "available_models": models,
        "current_model": {
            "model_name": current_model,
            "seed": current_seed
        } if current_model and current_seed else None
    }

@app.post("/set_model")
async def set_model(config: ModelConfig):
    """Set the model and seed to use for predictions"""
    global current_model, current_seed, classifier
    
    models = get_available_models()
    if config.model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model {config.model_name} not found")
    if config.seed not in models[config.model_name]:
        raise HTTPException(status_code=400, detail=f"Seed {config.seed} not found for model {config.model_name}")
    
    try:
        # Load the new model
        new_classifier = load_model(config.model_name, config.seed)
        
        # Update global variables only if model loading was successful
        current_model = config.model_name
        current_seed = config.seed
        classifier = new_classifier
        
        return {
            "message": "Model successfully set",
            "model_name": current_model,
            "seed": current_seed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.post("/predict")
async def predict(input_data: TextInput):
    """Make a prediction using the currently set model"""
    if not classifier or not current_model or not current_seed:
        raise HTTPException(
            status_code=400, 
            detail="No model is currently set. Please set a model using /set_model first"
        )
    
    try:
        prediction = classifier.predict(input_data.text)
        return {
            "prediction": prediction,
            "model": current_model,
            "seed": current_seed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
