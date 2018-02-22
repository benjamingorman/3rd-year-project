import json

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

def save_patterns_file(key_patterns, value_patterns, file_path):
    data = [[k,v] for (k,v) in zip(key_patterns, value_patterns)]
    with open(file_path, 'w') as f:
        f.write("[\n")
        for (i, pair) in enumerate(data):
            f.write(json.dumps(pair))
            if (i < len(data)-1):
                f.write(",")
            f.write("\n")
        f.write("]")

def load_patterns_file(file_path):
    with open(file_path, 'r') as f:
        patterns = json.load(f)

    assert(type(patterns) == list)
    assert(len(patterns) > 0)
    assert(type(patterns[0]) == list)
    return patterns

def create_cmm_input_file(key_patterns, value_patterns, file_path):
    key_size = len(key_patterns[0])
    value_size = len(value_patterns[0])
    with open(file_path, 'w') as f:
        f.write("{0}\n{1}\n".format(key_size, value_size))
        for (k,v) in zip(key_patterns, value_patterns):
            f.write(",".join(map(str, k)))
            f.write(":")
            f.write(",".join(map(str, v)))
            f.write("\n")
