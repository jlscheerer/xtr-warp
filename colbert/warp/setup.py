# NOTE It is IMPORTANT to import this file first to correctly configure CPU only execution!
import os
from dotenv import load_dotenv

# Ensure we are running use CPU only!
os.environ["CUDA_VISBLE_DEVICES"] = ""

# Load ENVIRONMENT variables. Be sure to change the .env file!
load_dotenv()

INDEX_ROOT = os.environ["INDEX_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

BEIR_COLLECTION_PATH = os.environ["BEIR_COLLECTION_PATH"]
LOTTE_COLLECTION_PATH = os.environ["LOTTE_COLLECTION_PATH"]
