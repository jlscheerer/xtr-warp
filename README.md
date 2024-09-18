# WARP: An Efficient Engine for Contextualized Multi-Vector Retrieval

------------

> WARP is an _extremely fast_ and _accurate_ retrieval engine based on [Stanford's ColBERTv2/PLAID](https://github.com/stanford-futuredata/ColBERT) and [Google DeepMind's XTR](https://github.com/google-deepmind/xtr) retrieval systems.

## Installation

WARP requires Python 3.7+ and PyTorch 1.9+ and uses the [Hugging Face Transformers](https://github.com/huggingface/transformers) library.
In addition, WARP supports the following runtimes for inference:
- [ONNX](https://onnx.ai/)
- [OpenVino](https://github.com/openvinotoolkit/openvino)
- [Core ML](https://developer.apple.com/documentation/coreml/) (macOS only)

It is strongly recommended to create a [conda environment](https://docs.anaconda.com/anaconda/install/linux/#installation) using the commands below.

We include an environment file specifically for CPU-only environments (`conda_env_cpu.yml`) and one for GPU environments (`conda_env.yml`).

```sh
conda env create -f conda_env[_cpu].yml
conda activate warp
```

> [!NOTE]
> While WARP's retrieval process is heavily optimized for CPU, it is still strongly recommended to use a GPU for index construction.

### Environment Setup
To construct indexes and perform retrieval, define the following environment variables in a `.env` file in the repository root:
```sh
INDEX_ROOT=...
EXPERIMENT_ROOT=...

BEIR_COLLECTION_PATH=...
LOTTE_COLLECTION_PATH=...
```

- `INDEX_ROOT`: Specifies the on-disk location for indexes.
- `EXPERIMENT_ROOT`: Specifies the on-disk location for experiment files.
- `BEIR_COLLECTION_PATH`: Designates the path to the datasets of the [BEIR Benchmark](https://github.com/beir-cellar/beir).
- `LOTTE_COLLECTION_PATH`: Specifies the path to the [LoTTE dataset](https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md).

### Dataset Setup

#### BEIR Benchmark

To download and extract a dataset from the BEIR Benchmark:

```sh
python utility/extract_collection.py -d ${dataset} -i "${BEIR_COLLECTION_PATH}" -s test
```

Replace `${dataset}` with the desired dataset name as specified [here](https://github.com/beir-cellar/beir?tab=readme-ov-file#beers-available-datasets).

#### LoTTE Dataset

1. Download the LoTTE dataset files from [here](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz).
2. Extract the files manually to the directory specified in `LOTTE_COLLECTION_PATH`.


> [!NOTE]
> If you face any problems, please feel free to [open a new issue](https://github.com/jlscheerer/xtr-warp/issues).

## Branches
- [`main`](https://github.com/jlscheerer/xtr-warptree/main): Stable branch with XTR/WARP.

## Bugs
If you experience bugs, or have suggestions for improvements, please use the issue tracker to report them.


------------

We provide code to reproduce the baseline evaluations for [XTR](https://github.com/jlscheerer/xtr-eval) and [ColBERTv2/PLAID](https://github.com/jlscheerer/colbert-eval).

> [!TIP]
> We provide scripts to reproduce all of these measurements on Google Cloud. The scripts can be found [here](https://github.com/jlscheerer/xtr-warp-gcp).
