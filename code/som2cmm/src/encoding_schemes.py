import numpy as np
from abc import ABC, abstractmethod
import math

def binomial(n, k):
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

class EncodingScheme(ABC):
    """An abstract base class for encoders"""

    @abstractmethod
    def encode(attrs):
        """Encode a list of attributes into a binary vector
        Args:
            attrs (list(float)): the list of attributes
        Returns:
            np.ndarray: the encoded vector
        """
        pass

    @abstractmethod
    def decode(vec):
        """Decode the given binary vector back into a list of attributes
        Args:
            vec (np.ndarray): the vector
        Returns:
            list(float): the list of attributes
        """
        pass

class QuantizationEncoder(EncodingScheme):
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

    def get_num_bins(self, bits_used, bits_set):
        return binomial(bits_used, bits_set)

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
        assert(len(code) == sum(self.bits_per_attr))
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
    pass

def new_quantize(bits_len, bits_set, bin_used):
    maxX = binomial(bits_len, bits_set)
    assert(bin_used < maxX)
    
    positions = []
    n = bits_len
    b = bits_set
    x = bin_used 

    for _ in range(bits_set):
        pos = get_first_bit_pos(n, b, x)
        #print(pos)
        positions.append(pos)

        for i in range(0, pos):
            x -= binomial(n-(i+1), b-1)

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
        y = binomial(n-1, b-1)
        while y < (x+1): 
            pos += 1
            y += binomial(n-(pos+1), b-1)
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
                y += binomial(n-(i+1), b-1)
    return y
