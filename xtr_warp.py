import os

# Ensure we are running use CPU only!
os.environ["CUDA_VISBLE_DEVICES"] = ""

from xtr_via_plaid import XTRRunConfig, to_colbert_config


def make_config(collection, datasplit, nbits, k):
    pass


if __name__ == "__main__":
    print("Hello World!")
