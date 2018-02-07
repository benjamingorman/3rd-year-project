import sys
import os.path
import random

num_patterns = 20
pattern_length = 5
pattern_min = 0.0
pattern_max = 1000.0

def gen_pattern():
    pat = []
    for _ in range(pattern_length):
        pat.append(random.random() * (pattern_max - pattern_min) - pattern_min)
    return pat

if __name__ == "__main__":
    out_path = sys.argv[1]
    assert(os.access(os.path.dirname(out_path), os.W_OK))

    with open(out_path, 'w') as f:
        for _ in range(num_patterns):
            f.write(",".join(map(str, gen_pattern())))
            f.write("\n")
