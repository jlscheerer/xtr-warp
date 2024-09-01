#!/usr/bin/bash
set -o allexport
source .env set
set +o allexport

# Prepare the BEIR datasets for evaluation
BEIR=("nfcorpus" "scifact" "scidocs" "fiqa" "webis-touche2020" "quora")
for dataset in "${BEIR[@]}"; do
    python utility/extract_collection.py -d ${dataset} -i "${BEIR_COLLECTION_PATH}" -s test
done

# Build Indexes for BEIR/test (nbits=4)
for dataset in "${BEIR[@]}"; do
    python utils.py index -c beir -d "$dataset" -s test -n 4
done

# Build Indexes for LoTTE.search/test (nbits=4)
LoTTE=("writing" "recreation" "science" "technology" "lifestyle" "pooled")
for dataset in "${LoTTE[@]}"; do
    python utils.py index -c lotte -d "$dataset" -t search -s test -n 4
done