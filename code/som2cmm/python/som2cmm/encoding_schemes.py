from abc import ABC, abstractmethod
import math
import numpy as np
import copy

from . import som
from . import utils

class EncodingScheme(ABC):
    """An abstract base class for encoders"""

    @abstractmethod
    def encode(self, attrs):
        """Encode a list of attributes into a binary vector
        Args:
            attrs (list(float)): the list of attributes
        Returns:
            np.ndarray: the encoded vector
        """
        pass

    def encodeAll(self, patterns):
        return [self.encode(pat) for pat in patterns]

    @abstractmethod
    def decode(self, vec):
        """Decode the given binary vector back into a list of attributes
        Args:
            vec (np.ndarray): the vector
        Returns:
            list(float): the list of attributes
        """
        pass

    def decodeAll(self, patterns):
        return [self.decode(pat) for pat in patterns]

    @abstractmethod
    def get_num_bits_in_encoding(self):
        """
        Returns:
            int: the number of bits set to 1 in  vectors encoded using the scheme
        """

class DoNothingEncoder(EncodingScheme):
    """Encoder which assumes it's input is already encoded"""

    def __init__(self, num_bits_in_encoding):
        self.num_bits_in_encoding = num_bits_in_encoding

    def get_num_bits_in_encoding(self):
        return self.num_bits_in_encoding

    def encode(self, attrs):
        return attrs

    def decode(self, vec):
        return vec

class QuantizationEncoder(EncodingScheme):
    """Encoder which performs quantization on each attribute"""

    def __init__(self, attr_min_max, bits_per_attr, bits_set_per_attr=2):
        """
        Args:
            attr_min_max (list(tuple(float, float))): the min and max values for each attribute
            bits_per_attr (int): how many bits to use per attribute
            bits_set_per_attr (int): how many bits to set per attribute
        """
        self.attr_min_max = attr_min_max
        self.bits_per_attr = bits_per_attr
        self.bits_set_per_attr = bits_set_per_attr

    def get_num_bits_in_encoding(self):
        return sum(self.bits_set_per_attr)

    def get_num_bins(self, bits_used, bits_set):
        return utils.binomial(bits_used, bits_set)

    def get_bin_size(self, min_val, max_val, num_bins):
        return (max_val - min_val) / float(num_bins)

    def get_bin_to_use(self, attr, min_val, bin_size, num_bins):
        b = math.floor((attr - min_val) / bin_size) 
        # Deals with max value
        if b == num_bins:
            b -= 1
        return b

    def encode(self, attrs):
        code = []

        for (i, attr) in enumerate(attrs):
            (min_val, max_val) = self.attr_min_max[i]
            bits_used = self.bits_per_attr[i]
            bits_set = self.bits_set_per_attr[i]
            attr_code = self.encode_attr(attr, min_val, max_val, bits_used, bits_set)

            for x in attr_code:
                code.append(x)

        return code

    def encode_attr(self, attr, min_val, max_val, bits_used, bits_set):
        """Quantize by initially having all the bits on the left and then shifting
        the rightmost one which is not at the end already
        So bin 0 looks like:
        1 1 1 0 0 0 0 0
        
        Then bin 1 looks like
        1 1 0 1 0 0 0 0
        ...
        Final bin looks like
        0 0 0 0 0 1 1 1
        """
        assert(min_val <= attr and attr <= max_val)
        assert(bits_used >= 1)
        assert(bits_set >= 1)
        assert(bits_set < bits_used)

        num_bins = self.get_num_bins(bits_used, bits_set)
        bin_size = self.get_bin_size(min_val, max_val, num_bins)
        bin_to_use = self.get_bin_to_use(attr, min_val, bin_size, num_bins)

        encoding = new_quantize(bits_used, bits_set, bin_to_use)
        return encoding

    def decode(self, code):
        try:
            assert(len(code) == sum(self.bits_per_attr))
        except Exception as e:
            print(code, self.bits_per_attr)
            raise e

        chunks = []
        ptr = 0

        for chunk_size in self.bits_per_attr: 
            chunks.append(code[ptr:ptr+chunk_size])
            ptr = ptr+chunk_size

        pattern = []
        for (i, chunk) in enumerate(chunks):
            (min_val, max_val) = self.attr_min_max[i]
            bits_used = self.bits_per_attr[i]
            bits_set = self.bits_set_per_attr[i]
            decoded_attr = self.decode_attr(chunk, min_val, max_val, bits_used, bits_set)
            pattern.append(decoded_attr)

        return pattern

    def decode_attr(self, code, min_val, max_val, bits_used, bits_set):
        bin_used = new_quantize_decode(bits_used, bits_set, code)

        num_bins = self.get_num_bins(bits_used, bits_set)
        bin_size = self.get_bin_size(min_val, max_val, num_bins)
        return min_val + bin_size * bin_used

class BaumEncoder(EncodingScheme):
    def __init__(self, segment_sizes):
        self.segment_sizes = segment_sizes
        self.last_baum_code_bits = None
        self.mappings = {}

    def get_num_bits_in_encoding(self):
        return len(self.segment_sizes)

    def encode(self, attrs):
        if len(self.mappings) == 0:
            baum_code_bits = get_initial_baum_code(self.segment_sizes)
        else:
            baum_code_bits = get_next_baum_code(self.segment_sizes, self.last_baum_code_bits) 

        self.last_baum_code_bits = copy.deepcopy(baum_code_bits)
        baum_code = concrete_baum_code(self.segment_sizes, baum_code_bits)
        self.mappings[repr(baum_code)] = repr(attrs)
        return baum_code

    def decode(self, vec):
        return self.mappings[repr(vec)]

class SOMEncoder(EncodingScheme):
    """Encoder which uses a Self-Organizing Map to perform the encoding"""

    def __init__(self, som_file_path):
        """
        Args:
            som_file_path(string): path to the .som file containing the trained som
        """
        self.som = som.SOM()
        self.som.loadFromFile(som_file_path)

    def get_num_bits_in_encoding(self):
        return 2

    def encode(self, vec):
        assert(len(vec) == self.som.numDims())
        (bmuRow, bmuCol) = self.som.findBMU(np.array(vec))

        # Divide the encoding into two groups of bits, one with a length equal to
        # the number of rows (1 bit set), and another with a length equal to
        # the number of cols (1 bit set).
        group1 = [0]*self.som.numRows()
        group2 = [0]*self.som.numCols()

        group1[bmuRow] = 1
        group2[bmuCol] = 1
        return group1 + group2

    def decode(self, vec):
        group1 = vec[:self.som.numRows()]
        group2 = vec[self.som.numRows():]
        row = group1.index(1)
        col = group2.index(1)
        bmuWeights = self.som.getNeuronWeights(row, col)
        return bmuWeights.tolist()

def get_initial_baum_code(segment_sizes):
    # returns bit positions
    bit_positions = []
    for _ in segment_sizes:
        bit_positions.append(0)
    return bit_positions

def get_next_baum_code(segment_sizes, bit_positions):
    for i in range(len(segment_sizes) - 1, -1, -1):
        if bit_positions[i] < segment_sizes[i] - 1:
            bit_positions[i] += 1
            break
        else:
            bit_positions[i] = 0
    return bit_positions

def concrete_baum_code(segment_sizes, bit_positions):
    code = []
    for (i, segsize) in enumerate(segment_sizes):
        seg = [0]*segsize
        seg[bit_positions[i]] = 1
        code += seg
    return code

def new_quantize(bits_len, bits_set, bin_used):
    maxX = utils.binomial(bits_len, bits_set)
    assert(bin_used < maxX)
    
    positions = []
    n = bits_len
    b = bits_set
    x = bin_used 

    for _ in range(bits_set):
        pos = get_first_bit_pos(n, b, x)
        positions.append(pos)

        for i in range(0, pos):
            x -= utils.binomial(n-(i+1), b-1)

        b -= 1
        n -= pos+1

    pat = [0] * bits_len
    cursor = 0
    for pos in positions:
        cursor += pos
        pat[cursor] = 1
        cursor += 1
    return pat

def get_first_bit_pos(n, b, x):
    if b == 1:
        return x
    else:
        pos = 0
        y = utils.binomial(n-1, b-1)
        while y < (x+1): 
            pos += 1
            y += utils.binomial(n-(pos+1), b-1)
        return pos

def new_quantize_decode(bits_len, bits_used, code):
    y = 0
    n = bits_len
    b = bits_used
    for (i, v) in enumerate(code):
        if b == 1:
            y += code[i:].index(1)
            break
        else:
            if v == 1:
                b -= 1
            else:
                y += utils.binomial(n-(i+1), b-1)
    return y
