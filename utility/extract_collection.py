import os
import argparse
import json

from beir import util
from beir.datasets.data_loader import GenericDataLoader

def extract_collection_beir(dataset, input_path, output_path, split):
    if not os.path.exists(os.path.join(input_path, dataset)):
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        data_path = util.download_and_unzip(url, input_path)
        print(f"#> Extracted BEIR dataset to {data_path}")

    input_path = os.path.join(input_path, dataset)
    output_path = os.path.join(output_path, dataset)

    corpus, queries, qrels = GenericDataLoader(input_path).load(split=split)

    collection, collection_map = [], dict()
    for line_num, (id_, document) in enumerate(corpus.items()):
        title, text = document["title"], document["text"]

        # Escape newline characters in the title/text
        title = title.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

        # NOTE We are using the "XTR way" of concatenating title and text
        # https://colab.research.google.com/github/google-deepmind/xtr/blob/main/xtr_evaluation_on_beir_miracl.ipynb
        collection.append(f"{line_num}\t{title} {text}\n")
        collection_map[line_num] = id_

    collection_path = os.path.join(output_path, "collection.tsv")
    with open(collection_path, "w") as file:
        file.writelines(collection)

    collection_map_path = os.path.join(output_path, "collection_map.json")
    with open(collection_map_path, "w") as file:
        file.write(json.dumps(collection_map))

    questions = []
    for id_, query in queries.items():
        questions.append(f"{id_}\t{query}\n")

    questions_file = os.path.join(output_path, f"questions.{split}.tsv")
    with open(questions_file, "w") as file:
        file.writelines(questions)

    qrels_file = os.path.join(output_path, f"qrels.{split}.json")
    with open(qrels_file, "w") as file:
        json.dump(qrels, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("extract_collection.py")
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-s", "--split", required=True)
    parser.add_argument("-o", "--output")

    args = parser.parse_args()
    extract_collection_beir(args.dataset, args.input, args.output or args.input, split=args.split)