import os
import argparse
import json
import jsonlines

def extract_collection(input_path, output_path):
    corpus_file = os.path.join(input_path, "corpus.jsonl")

    collection, collection_map = [], dict()
    with jsonlines.open(corpus_file, "r") as file:
        for line_num, line in enumerate(file):
            id_, title, text = line["_id"], line["title"], line["text"]
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

    def extract_question_ids(filename):
        question_ids = set()
        qrels_path = os.path.join(input_path, "qrels")
        with open(os.path.join(qrels_path, filename)) as file:
            lines = [line.strip().split("\t") for line in file]
            header, *data = lines
            assert header == ["query-id", "corpus-id", "score"]
            for query_id, corpus_id, score in data:
                question_ids.add(int(query_id))
        return question_ids

    train_questions, test_questions = extract_question_ids("train.tsv"), extract_question_ids("test.tsv")

    queries_file = os.path.join(output_path, "queries.jsonl")

    # TODO(jlscheerer) split questions into test/train based on qrels.
    questions_train, questions_test = [], []
    with jsonlines.open(queries_file, "r") as file:
        for line in file:
            id_, text = int(line["_id"]), line["text"]
            formatted_line = f"{id_}\t{text}\n"
            if id_ in train_questions:
                questions_train.append(formatted_line)
            else:
                assert id_ in test_questions
                questions_test.append(formatted_line)
    
    questions_file = os.path.join(output_path, "questions.train.tsv")
    with open(questions_file, "w") as file:
        file.writelines(questions_train)

    questions_file = os.path.join(output_path, "questions.test.tsv")
    with open(questions_file, "w") as file:
        file.writelines(questions_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("extract_collection.py")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output")

    args = parser.parse_args()
    extract_collection(args.input, args.output or args.input)