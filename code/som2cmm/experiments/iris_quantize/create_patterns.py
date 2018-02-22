import json

path = "/home/userfs/b/bg739/modules/3rd-year-project/code/som2cmm/data/iris/iris.data.original"

def encode_iris_class(cls):
    if cls == "Iris-setosa":
        return [1, 0, 0]
    elif cls == "Iris-versicolor":
        return [0, 1, 0]
    elif cls == "Iris-virginica":
        return [0, 0, 1]
    else:
        return cls

keys = []
values = []
with open(path, 'r') as f:
    for line in f:
        line = line.strip()
        if len(line) == 0:
            continue
        parts = line.split(",")
        attrs = parts[:4]
        iris_class = parts[4]

        keys.append(list(map(float, attrs)))
        values.append(encode_iris_class(iris_class))

data = list(zip(keys, values))
with open("iris_patterns.json", 'w') as f:
    json.dump(data, f, indent=4)
