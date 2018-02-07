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

        #print("bin_to_use", bin_to_use, "attr", attr, "min val", min_val, "bin_size", bin_size)
        for (i, code) in enumerate(self.enumerate_encodings(bits_used, bits_set)):
            if i == bin_to_use:
                return code

    def enumerate_encodings(self, bits_used, bits_set):
        num_encodings = self.get_num_bins(bits_used, bits_set)
        bit_positions = list(range(bits_set))
        code = [0] * bits_used

        max_bit = bits_set - 1
        max_pos = len(code) - 1

        for n in range(num_encodings):
            #print("encoding " + str(n))
            for i in range(len(code)):
                code[i] = 0
            for pos in bit_positions:
                code[pos] = 1

            #print("code", code)
            
            yield code
            if n == num_encodings - 1: 
                break

            # figure out which bit to move
            rightmost_movable_bit = None
            for i in reversed(range(bits_set)):
                pos = bit_positions[i]

                if i == max_bit:
                    if pos < max_pos:
                        rightmost_movable_bit = i
                        break
                elif bit_positions[i+1] != pos+1:
                    rightmost_movable_bit = i
                    break

            # This would only occur if bin_to_use > num_bins
            assert(rightmost_movable_bit != None)
            bit_positions[rightmost_movable_bit] += 1

            # Move back all bits ahead
            for i in reversed(range(bits_set)):
                if i > rightmost_movable_bit:
                    bit_positions[i] = bit_positions[rightmost_movable_bit] + (i-rightmost_movable_bit)

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
        bin_used = None
        for (i, other_code) in enumerate(self.enumerate_encodings(bits_used, bits_set)):
            if code == other_code:
                bin_used = i

        assert(bin_used != None)

        num_bins = self.get_num_bins(bits_used, bits_set)
        bin_size = self.get_bin_size(min_val, max_val, num_bins)
        return min_val + bin_size * bin_used
