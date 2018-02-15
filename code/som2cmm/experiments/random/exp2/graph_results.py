import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import json
import os

DIR = os.path.dirname(os.path.realpath(__file__))

X = []
Y = []
Z = []
for item in os.listdir(DIR): 
    if os.path.isdir(item) and os.path.isfile(item + "/stats.json"):
        print(item)
        bits_per_attr, bits_set_per_attr = map(int, item.split("-"))
        stats = json.load(open(item + "/stats.json"))
        acc = stats["correct"] / 1000.0

        print(bits_per_attr, bits_set_per_attr, acc)
        X.append(bits_per_attr)
        Y.append(bits_set_per_attr)
        Z.append(acc)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("Bits per attr")
ax.set_ylabel("Bits set per attr")
ax.set_zlabel("Accuraccy")

surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
