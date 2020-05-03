import argparse
import textgrid

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--path')
parser.add_argument('--phoneme_tier', default="PHONEMES")
parser.add_argument('--sr', default=16000)
args = parser.parse_args()

out = args.path.replace(".TextGrid", ".phn")
g = textgrid.TextGrid()
g.read(args.path)
phonemes = g.getList(args.phoneme_tier)[0]

with open(out, "w") as f:
    for phn in phonemes:
        phn_name = phn.mark
        phn_start = int(phn.minTime * args.sr)
        phn_end = int(phn.maxTime * args.sr)
        f.write(f"{phn_start} {phn_end} {phn_name}\n") 