import os
from dotenv import load_dotenv

# Ensure we are running use CPU only!
os.environ["CUDA_VISBLE_DEVICES"] = ""

# Load ENVIRONMENT variables. Be sure to change the .env file!
load_dotenv()

from xtr_via_plaid import XTRRunConfig, to_colbert_config


def make_config(collection, datasplit, nbits, k):
    pass


if __name__ == "__main__":
    print(os.environ["INDEX_ROOT"])
