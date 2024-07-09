import os
from dotenv import load_dotenv

# Ensure we are running use CPU only!
os.environ["CUDA_VISBLE_DEVICES"] = ""

# Load ENVIRONMENT variables. Be sure to change the .env file!
load_dotenv()

from colbert.xtr_run_config import XTRRunConfig


if __name__ == "__main__":
    config = XTRRunConfig(
        nranks=4,
        dataset="lotte",
        collection="writing",
        type_="search",
        datasplit="test",
        nbits=4,
        k=100,
    )

    print(config.index_root)
