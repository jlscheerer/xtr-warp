import sys
import subprocess

if __name__ == "__main__":
    collections = ["lifestyle", "pooled", "recreation", "science", "technology", "writing"]
    datasplits = ["dev", "test"]
    nbits = [1, 2, 4]
    for collection in collections:
        for split in datasplits:
            for nbit in nbits:
                subprocess.run(["python", "xtr_via_plaid.py", "convert", "-c", collection, "-s", split, "-n", f"{nbit}"], 
                                stderr=sys.stderr, stdout=sys.stdout)
