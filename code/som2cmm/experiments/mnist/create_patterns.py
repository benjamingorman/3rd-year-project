import json

path = "mnist_som_test_1000.txt"

def encode_class(cls):
    encoding = [0,0,0, 0,0,0, 0,0,0, 0]
    cls = int(cls)
    encoding[cls] = 1
    return encoding

keys = []
values = []
with open(path, 'r') as f:
    i = 0
    for line in f:
        i += 1
        if i == 1001:
            break
        line = line.strip()
        if len(line) == 0:
            continue
        parts = line.split(",")
        attrs = parts[:-1]
        cls = parts[-1]

        keys.append(list(map(float, attrs)))
        values.append(encode_class(cls))

data = list(zip(keys, values))
with open("mnist_patterns_1000.json", 'w') as f:
    json.dump(data, f, indent=None)
