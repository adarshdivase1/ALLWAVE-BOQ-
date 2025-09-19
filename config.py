import os
from typing import Dict, Any

class Config:
    # Hugging Face Configuration
    HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
    HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    # AVIXA Guidelines
    AVIXA_STANDARDS = {
        "display_distance": {
            "min_ratio": 6,  # H:W ratio
            "max_ratio": 10
        },
        "audio_coverage": {
            "spl_target": 70,  # dB
            "frequency_response": "20Hz-20kHz"
        },
        "control_redundancy": True,
        "cable_management": "TIA-568",
        "room_acoustics": {
            "rt60_target": 0.6,  # seconds
            "noise_floor": 35   # dB
        }
    }
    
    # Model Configuration
    MODEL_CONFIG = {
        "embedding_dim": 384,
        "max_sequence_length": 512,
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 50
    }
    
    # File Paths
    DATA_DIR = "data"
    MODELS_DIR = "models"
    REPORTS_DIR = "reports"
    
    @classmethod
    def get_avixa_guidelines(cls) -> Dict[str, Any]:
        return cls.AVIXA_STANDARDS
