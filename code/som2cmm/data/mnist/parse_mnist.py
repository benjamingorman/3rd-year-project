import re

with open("mnist_test_1000.txt", 'r') as f:
    with open("mnist_som_test_1000.txt", 'w') as output_f:
        i = 0
        for line in f:
            match = re.match(r"^\|labels (.*) \|features (.*)$", line)
            label_code = match.group(1)
            features_code = match.group(2)
            label = label_code.split(" ").index('1')

            output_line = features_code.split(" ")
            output_line.append(str(label))
            output_f.write(",".join(output_line))
            output_f.write("\n")
