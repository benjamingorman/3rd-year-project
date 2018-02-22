import sys
import os.path
import random
import argparse
import json

def gen_pattern(plen, pmin, pmax):
    pat = []
    for _ in range(plen):
        n = random.random() * (pmax - pmin) - pmin
        pat.append(round(n, 3))
    return pat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Path to the output file")
    parser.add_argument("--n", required=True, type=int, help="Number of patterns")
    parser.add_argument("--len", required=True, type=int, help="Pattern length")
    parser.add_argument("--min", required=True, type=float, help="Pattern min value")
    parser.add_argument("--max", required=True, type=float, help="Pattern max value")
    args = parser.parse_args()

    assert(os.access(os.path.dirname(args.out), os.W_OK))

    key_pats = []
    value_pats = []
    for _ in range(args.n):
        key_pats.append(gen_pattern(args.len, args.min, args.max))
        value_pats.append(gen_pattern(args.len, args.min, args.max))

    data = [list(pair) for pair in zip(key_pats, value_pats)]

    with open(args.out, 'w') as f:
        json.dump(data, f, indent=4)
