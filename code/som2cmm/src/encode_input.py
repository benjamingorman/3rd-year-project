from . import encoding_schemes as enc
import argparse
import os
import os.path

def load_patterns(input_path, class_index=None):
    patterns = []
    pattern_length = None # use the length of the first pattern to configure pattern length

    with open(input_path, 'r') as f:
        for line in f:
            if len(line) == 0:
                continue

            pattern = []
            items = line.split(",")
            for i in range(len(items)):
                if i == class_index:
                    continue
                else:
                    pattern.append(float(items[i]))

            patterns.append(pattern)
            if pattern_length != None:
                # Ensure all patterns same size
                assert(len(pattern) == pattern_length)
            else:
                pattern_length = len(pattern)

    return patterns

def get_min_max_values(patterns):
    pattern_length = len(patterns[0])
    vmin = float("inf")
    vmax = float("-inf")
    min_values = [vmin] * pattern_length
    max_values = [vmax] * pattern_length

    for pat in patterns:
        for i in range(pattern_length):
            if pat[i] < min_values[i]:
                min_values[i] = pat[i]
            if pat[i] > max_values[i]:
                max_values[i] = pat[i]
    
    return list(zip(min_values, max_values))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the input file")
    parser.add_argument("--output", required=True, help="Path to the output file")
    parser.add_argument("--input-file-class-index", type=int, help="Index of the pattern class in the input file. If given then this index will be ignored when constructing the pattern")
    parser.add_argument("--encoding", required=True, help="Which encoding to use",
                        choices=["quantize"])
    parser.add_argument("--quantize-bits-per-attr", type=int, nargs='+', help="A list of how many bits to use for each attribute")
    parser.add_argument("--quantize-bits-set-per-attr", type=int, nargs='+', help="A list of how many bits to set for each attribute")
    args = parser.parse_args()

    assert(os.path.isfile(args.input))
    assert(os.access(os.path.dirname(args.output), os.W_OK))

    patterns = load_patterns(args.input, class_index=args.input_file_class_index)
    pattern_length = len(patterns[0])
    min_max_values = get_min_max_values(patterns)

    encoded_patterns = []

    if args.encoding == "quantize":
        assert(len(args.quantize_bits_per_attr) == pattern_length)
        assert(len(args.quantize_bits_set_per_attr) == pattern_length)

        encoder = enc.QuantizationEncoder(min_max_values,
                args.quantize_bits_per_attr, args.quantize_bits_set_per_attr)

        for pat in patterns:
            encoded_patterns.append(encoder.encode(pat))

    with open(args.output, 'w') as f:
        for pat in encoded_patterns:
            f.write(",".join(map(str, pat)))
            f.write("\n")

