import argparse
import os
import os.path

from . import encoding_schemes as schemes
from . import utils

def get_scheme(patterns, config):
    """Returns the encoding scheme specified by the given config object
    Args:
        patterns (list(list)): List of input patterns
        config (dict): The config object
    """
    assert(type(patterns) == list and len(patterns) > 0)
    assert(type(config) == dict)

    min_max_values = utils.get_min_max_values(patterns)
    pattern_dims = len(patterns[0])

    scheme = None
    method = config["method"]

    if method == "quantize":
        bits_per_attr = config["quantize"]["bits_per_attr"]
        bits_set_per_attr = config["quantize"]["bits_set_per_attr"]
        assert(type(bits_per_attr) == list and len(bits_per_attr) == pattern_dims)
        assert(type(bits_set_per_attr) == list and len(bits_set_per_attr) == pattern_dims)
        scheme = schemes.QuantizationEncoder(min_max_values, bits_per_attr, bits_set_per_attr)
    elif method == "donothing":
        bits_set = config["donothing"]["bits_set"]
        scheme = schemes.DoNothingEncoder(bits_set)
    elif method == "som":
        som_path = config["som"]["som_file_path"]
        scheme = schemes.SOMEncoder(som_path)
    else:
        raise ValueError("Unrecognized encoding method: " + method)

    return scheme

def load_patterns_from_file(input_path, class_index=None):
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

"""
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
    min_max_values = utils.get_min_max_values(patterns)

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
"""
