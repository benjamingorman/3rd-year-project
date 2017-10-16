import numpy 


def create_matrix(rows, cols):
    return numpy.zeros(shape=(rows, cols))

def create_vector(size):
    return create_matrix(size, 1)

class CMM:
    """
    Represents a correlation matrix memory.
    """
    def __init__(self, rows, cols):
        self._data = create_matrix(rows, cols)

    def __str__(self):
        return str(self._data)

    def num_rows(self):
        return self._data.shape[0]

    def num_cols(self):
        return self._data.shape[1]

    def insert(self, key_vec, data_vec):
        """
        Insert a new item into the memory.
        @param key_vec the key vector
        @param data_vec the data vector
        """
        # Ensure correctly sized vectors
        assert(key_vec.shape[0] == self.num_rows())
        assert(key_vec.shape[1] == 1)

        assert(data_vec.shape[0] == self.num_cols())
        assert(data_vec.shape[1] == 1)

        # Temporary matrix m will be added to the main matrix
        m = create_matrix(self.num_rows(), self.num_cols())

        numpy.dot(data_vec, numpy.transpose(key_vec), out=m)

        self._data += m

    def recall(self, key_vec):
        """
        Recalls an item from the memory.
        @param key_vec the key vector
        """
        output = create_vector(self.num_cols())
        numpy.dot(self._data, key_vec, out=output)
        return output

def parse_input_file(filename):
    print("Parsing input file: " + filename)
    key_size = 0
    data_size = 0
    pairs = []
    with open(filename, 'r') as f:
        for n, line in enumerate(f):
            # First line is key vector size
            if n == 0:
                key_size = int(line.strip())
            # Second line is data vector size
            elif n == 1:
                data_size = int(line.strip())
            # Other lines hold key and data vector pairs
            else:
                key_vec = create_vector(key_size)
                data_vec = create_vector(data_size)

                key_vec_str, data_vec_str = line.strip().split(":")
                for i, x in enumerate(key_vec_str.split(",")):
                    key_vec[i, 0] = float(x)

                for i, x in enumerate(data_vec_str.split(",")):
                    data_vec[i, 0] = float(x)

                pairs.append((key_vec, data_vec))
    return key_size, data_size, pairs


def run_input_file(filename):
    key_size, data_size, pairs = parse_input_file(filename)
    print("Key size", key_size)
    print("Data size", data_size)

    mem = CMM(key_size, data_size)
    for key_vec, data_vec in pairs:
        print("*** Inserting...")
        print("Key vec:")
        print(key_vec)
        print("Data vec:")
        print(data_vec)
        mem.insert(key_vec, data_vec)
        print("Memory:")
        print(mem)

    print("\n*** Recalling...\n")

    for key_vec, data_vec in pairs:
        print("Key vec:")
        print(key_vec)
        print("Data vec (recalled):")
        print(mem.recall(key_vec))
        print("Data vec (original):")
        print(data_vec)


def quick_test():
    mem = CMM(3,3)

    key_vec = create_vector(3)
    key_vec[0,0] = 1 
    key_vec[1,0] = 2 
    key_vec[2,0] = 3 

    data_vec = create_vector(3)
    data_vec[0,0] = 4 
    data_vec[1,0] = 5 
    data_vec[2,0] = 6 

    mem.insert(key_vec, data_vec)

    print("Key vector:")
    print(key_vec)
    print("Data vector:")
    print(data_vec)
    print("CMM")
    print(mem)
    print("Recalled vector")
    print(mem.recall(key_vec))

if __name__ == "__main__":
    run_input_file("cmm_input.txt")
