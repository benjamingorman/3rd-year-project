import numpy as np
import sys
import json
import argparse
import os
import os.path


def create_matrix(rows, cols):
    return np.zeros(shape=(rows, cols))

def create_vector(size):
    return create_matrix(size, 1)

def set_gt1_to_1(mat):
    mat[mat > 1] = 1

def binary_vec_to_str(vec, vertical=False):
    elements = map(lambda x: str(int(x)), vec[:, 0])
    if vertical:
        return "\n".join(elements)
    else:
        return "".join(elements)

def binary_mat_to_str(mat):
    row_strings = []
    rows, cols = mat.shape
    for row in range(rows):
        row_str = "".join(map(lambda x: str(int(x)), mat[row, :]))
        row_strings.append(row_str)
    return "\n".join(row_strings)

def argmin(func, xs):
    min_x = None
    min_value = float("inf")

    for x in xs:
        value = func(x)
        if value < min_value:
            min_x = x
            min_value = value

    return min_x

def vector_distance(v1, v2):
    return np.linalg.norm(np.subtract(v1, v2))

def mean(xs):
    return sum(xs) / float(len(xs))

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class CMM:
    """Represents a correlation matrix memory."""

    def __init__(self, key_size, data_size, bits_in_key, record_data_items=True):
        """Note that the matrix is shaped with `data_size` rows and `key_size` cols
        e.g.
              k k k k 
            d 0 0 0 0
            d 0 0 0 0
            d 0 0 0 0
        Args:
            key_size (int): The size of the key vector
            data_size (int): The size of the data vector
            record_data_items (bool):
                If true, every input data vector will be kept so that when recalling
                we can match to the closest input data vector.
        """
        self._mat = create_matrix(data_size, key_size)
        # This saves having to allocate a new matrix on every insert
        self._work_mat = create_matrix(data_size, key_size)
        self.bits_in_key = bits_in_key
        self.should_record_data_items = record_data_items
        self.recorded_data_items = []

    def __str__(self):
        return self.print_mem()

    def key_size(self):
        return self.num_cols()

    def data_size(self):
        return self.num_rows()

    def num_rows(self):
        return self._mat.shape[0]

    def num_cols(self):
        return self._mat.shape[1]

    def serialize_mem(self):
        return binary_mat_to_str(self._mat)

    def print_mem(self):
        print(self.serialize_mem())

    def insert(self, key_vec, data_vec):
        """Insert a new item into the memory.
        Args:
            key_vec (np.ndarray): the key vector
            data_vec (np.ndarray): the data vector
        """
        # Ensure correctly sized vectors
        assert(data_vec.shape[0] == self.num_rows())
        assert(data_vec.shape[1] == 1)

        assert(key_vec.shape[0] == self.num_cols())
        assert(key_vec.shape[1] == 1)

        assert(np.count_nonzero(key_vec) == self.bits_in_key)

        # Temporary matrix will be added to the main matrix
        self._work_mat.fill(0)
        np.dot(data_vec, np.transpose(key_vec), out=self._work_mat)
        self._mat += self._work_mat
        set_gt1_to_1(self._mat)

        if self.should_record_data_items:
            self.recorded_data_items.append(data_vec)

    def recall_basic(self, key_vec):
        """Recalls an item from the memory.
        Args:
            key_vec (np.ndarray): the key vector
        Returns:
            np.ndarray: the recalled data vector
        """
        output_vec = create_vector(self.data_size())
        output_vec.fill(0)
        np.dot(self._mat, key_vec, out=output_vec)
        return self.threshold(output_vec)

    def threshold(self, vec):
        bits = self.bits_in_key

        indices_w_values = []
        for (i, row) in enumerate(vec):
            indices_w_values.append((i, row[0]))

        # Sort by value
        indices_w_values.sort(key=lambda x: -x[1])
        top = indices_w_values[:bits]

        new_vec = create_vector(len(vec))
        for (i, _) in top:
            new_vec[i][0] = 1

        return vec


    def recall_smart(self, key_vec):
        """Peforms a basic recall and then attempts to match the result against
        the list of stored data items to find the closest one, then returns that.
        """
        assert(self.should_record_data_items and len(self.recorded_data_items) > 0)
        basic_result = self.recall_basic(key_vec)
        closest_match = argmin(lambda v: vector_distance(v, basic_result), self.recorded_data_items)
        return closest_match

    def recall(self, key_vec, smart=True):
        if smart:
            return self.recall_smart(key_vec)
        else:
            return self.recall_basic(key_vec)

def parse_input_file(filename):
    print("Parsing input file: " + filename)
    key_size = 0
    data_size = 0
    pairs = []
    with open(filename, 'r') as f:
        for (n, line) in enumerate(f):
            if n == 0:
                # First line is key vector size
                key_size = int(line.strip())
            elif n == 1:
                # Second line is data vector size
                data_size = int(line.strip())

            # Other lines hold key and data vector pairs
            else:
                key_vec = create_vector(key_size)
                data_vec = create_vector(data_size)

                key_vec_str, data_vec_str = line.strip().split(":")
                for (i, x) in enumerate(key_vec_str.split(",")):
                    key_vec[i, 0] = float(x)

                for (i, x) in enumerate(data_vec_str.split(",")):
                    data_vec[i, 0] = float(x)

                pairs.append((key_vec, data_vec))
    return key_size, data_size, pairs

def save_output_file(output_path, results):
    """Saves the results to the given file.
    Args:
        output_path (str): The path to the output file
        results (tuple(np.ndarray, np.ndarray, np.ndarray)): The list of results
    """
    with open(output_path, 'w') as f:
        for (i, (key_vec, data_vec, data_recalled)) in enumerate(results):
            f.write("key {}".format(i).ljust(9))
            f.write(binary_vec_to_str(key_vec))
            f.write("\n")
            f.write("original ")
            f.write(binary_vec_to_str(data_vec))
            f.write("\n")
            f.write("recalled ")
            f.write(binary_vec_to_str(data_recalled))
            f.write("\n\n")

def save_stats_file(output_path, results):
    """Saves statistics about the results to the given file
    Args:
        output_path (str): The path to the output file
        results (tuple(np.ndarray, np.ndarray, np.ndarray)): The list of results
    """
    stats = {}
    stats["correct"] = 0
    stats["wrong"] = 0

    bits_wrong = []
    for (key_vec, data_vec, data_recalled) in results:
        if (data_vec == data_recalled).all():
            stats["correct"] += 1
        else:
            stats["wrong"] += 1
        bits_wrong.append(vector_distance(data_vec, data_recalled))
    stats["mean bits wrong"] = mean(bits_wrong)
    stats["stdev bits wrong"] = np.std(bits_wrong)

    with open(output_path, 'w') as f:
        f.write(json.dumps(stats, indent=4, sort_keys=True))

def debug_result(result):
    (key_vec, data_vec, data_recalled) = result
    print("key:      {}".format(binary_vec_to_str(key_vec)))
    print("data:     {}".format(binary_vec_to_str(data_vec)))
    print("recalled: {}".format(binary_vec_to_str(data_recalled)))

def run_experiment(input_path, out_dir_path, bits_in_key, use_smart_recall=True):
    key_size, data_size, pairs = parse_input_file(input_path)
    print("*** Running experiment: {}".format(input_path))
    print("Output directory: {}".format(out_dir_path))
    print("Key size: {}".format(key_size))
    print("Bits in key {}".format(bits_in_key))
    print("Data size: {}".format(data_size))
    print("Using smart recall: {}".format(use_smart_recall))

    cmm = CMM(key_size, data_size, bits_in_key)

    print("")
    print("* Training...")
    for key_vec, data_vec in pairs:
        cmm.insert(key_vec, data_vec)

    print("Training complete.")

    print("")
    print("* Recalling...")
    results = []
    for key_vec, data_vec in pairs:
        data_recalled = cmm.recall(key_vec, smart=use_smart_recall)
        result = (key_vec, data_vec, data_recalled)
        results.append(result)

    print("Recall complete.")

    print("")
    mem_file = os.path.join(out_dir_path, "cmm.txt")
    results_file = os.path.join(out_dir_path, "results.txt")
    stats_file = os.path.join(out_dir_path, "stats.json")

    print("Memory file:  {}".format(mem_file))
    print("Results file: {}".format(results_file))
    print("Stats file:   {}".format(stats_file))

    with open(mem_file, 'w') as f:
        f.write(cmm.serialize_mem())

    save_output_file(results_file, results)
    save_stats_file(stats_file, results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument("--input", required=True, help="Path to the input file")
    parser.add_argument("--bits-in-key", required=True, type=int, help="How many bits are set in the key patterns")
    parser.add_argument("--out-dir", required=True, help="Path to the output directory (will be created if it doesn't exist)")
    parser.add_argument("--smart-recall", type="bool", default=True, help="Use smart recall?")
    args = parser.parse_args()

    assert(os.path.isfile(args.input))
    if not os.path.isdir(args.out_dir):
        print("Creating directory: " + args.out_dir)
        os.mkdir(args.out_dir)

    run_experiment(args.input, args.out_dir, args.bits_in_key, use_smart_recall=args.smart_recall)
