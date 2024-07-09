import os
import json

import torch
import numpy as np

from tqdm import tqdm

from colbert.modeling.xtr import DOC_MAXLEN, QUERY_MAXLEN

if __name__ == "__main__":
    print(DOC_MAXLEN)
