import sys
import json

json_file = sys.argv[1]
out_file = sys.argv[2]
print(json_file)
print(out_file)

with open(json_file, 'r') as f:
    data = json.load(f)

keys = [pair[1] for pair in data]
with open(out_file, 'w') as f:
    for k in keys:
        f.write(",".join(map(str, k)))
        f.write("\n")
