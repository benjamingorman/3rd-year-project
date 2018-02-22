import numpy as np

class SOM:
    """Represents a Self-Organizing Map. Can only be used to execute pre-trained
    SOMs at this time. For training SOMs use som.c
    """
    def __init__(self):
        self.data = None

    def loadFromFile(self, file_path):
        with open(file_path, 'r') as f:
            rows = 0
            cols = 0
            dims = 0
            count = 0 # check to make sure the file is valid
            for (i, line) in enumerate(f):
                if i == 0:
                    rows, cols, dims = map(int, line.split(","))
                    self.data = np.zeros(shape=(rows,cols,dims))
                else:
                    weights = list(map(float, line.split(",")))
                    row = int(count / float(cols))
                    col = count % cols
                    self.data[row,col] = np.array(weights)
                    count += 1
            assert(count == rows * cols)

    def getNeuronWeights(self, row, col):
        return self.data[row, col, :]

    def numRows(self):
        return self.data.shape[0]

    def numCols(self):
        return self.data.shape[1]

    def numDims(self):
        return self.data.shape[2]

    def distanceFunc(self, vec1, vec2):
        return np.linalg.norm(vec2 - vec1)

    def findBMU(self, pattern):
        """Finds the best matching unit to the given pattern
        Args:
            pattern (np.array): The input pattern as a vector of float values
        """
        assert(len(pattern) == self.numDims())
        bestUnit = (0, 0)
        bestDist = float("inf")
        for col in range(self.numCols()):
            for row in range(self.numRows()):
                weights = self.getNeuronWeights(row, col)
                dist = self.distanceFunc(pattern, weights)
                if dist < bestDist:
                    bestUnit = (row, col)
                    bestDist = dist 
        return bestUnit
