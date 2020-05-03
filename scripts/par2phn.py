import argparse

parser = argparse.ArgumentParser(description="convert .par format from phonedat1 dataset (/data/felix/datasets/phonedat1)")
parser.add_argument('--path', help="path to a .par file, the output .phn file will be in the same location")
args = parser.parse_args()
out = args.path.replace(".par", ".phn")

with open(args.path, "r") as src:
    with open(out, "w") as trg:
        lines = src.readlines()
        lines = list(filter(lambda x: 'MAU' in x, lines))

        for line in lines:
            split_line = line.split("\t")
            start, length, phn = int(split_line[1]), int(split_line[2]), split_line[3]
            trg.write(f"{start} {start + length} {phn}\n") 