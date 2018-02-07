import random
import sys

patterns = 5 
key_size = 20
data_size = 20
key_bits = 1
data_bits = 1
output_path = "output/random_cmm_input.txt"

def gen_vec(length, bits):
    xs = [0 for _ in range(length)]
    indices = random.sample(range(0, length), bits)
    for i in indices:
        xs[i] = 1
    return xs

def gen_key_vec():
    return gen_vec(key_size, key_bits)

def gen_data_vec():
    return gen_vec(data_size, data_bits)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    print("output_path", output_path)

    with open(output_path, 'w') as f:
        f.write(str(key_size))
        f.write("\n")
        f.write(str(data_size))
        f.write("\n")

        for _ in range(patterns):
            f.write(",".join(map(str, gen_key_vec())))
            f.write(":")
            f.write(",".join(map(str, gen_data_vec())))
            f.write("\n")
