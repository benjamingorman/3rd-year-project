import math
import numpy as np
import sys
import random
import svgwrite
from PIL import Image

LATTICE_WIDTH = 20
LATTICE_HEIGHT = 20
INITIAL_LEARNING_RATE = 0.1
INPUT_DIMENSIONS = 3
WEIGHTS = np.random.rand(LATTICE_HEIGHT, LATTICE_WIDTH, INPUT_DIMENSIONS)
WEIGHTS = WEIGHTS * 50 + 100 # to better approximate RGB colour values
#WEIGHTS = WEIGHTS * 0
DELTA0 = 10.0
NEIGHBOURHOOD_SHRINK_FACTOR = 600.0
LEARNING_RATE_SHRINK_FACTOR = 300.0
EPOCH_SIZE = 1
NUM_EPOCHS = 1000

def negexp(x):
    return math.exp(-x)

def discriminant_function(w1, w2):
    return np.linalg.norm(w2 - w1)

def neighbourhood_size(time):
    #return DELTA0 * (1 - math.exp(-time / NEIGHBOURHOOD_SHRINK_FACTOR))
    return DELTA0 * negexp(time / NEIGHBOURHOOD_SHRINK_FACTOR)

def learning_rate(time):
    return INITIAL_LEARNING_RATE * math.exp(-time / LEARNING_RATE_SHRINK_FACTOR)

def neighbourhood_function(neuron1, neuron2, time):
    (x1, y1) = neuron1
    (x2, y2) = neuron2
    dist = ((x2 - x1)**2 + (y2 - y1) ** 2)
    #T = negexp(dist / 2 * neighbourhood_size(time)**2)
    T = negexp(dist / neighbourhood_size(time))
    return T

def find_winning_neuron(pattern):
    best_d = sys.float_info.max
    best_neuron = (0, 0)
    for y in range(LATTICE_HEIGHT):
        for x in range(LATTICE_WIDTH):
            neuron_weights = WEIGHTS[y][x]
            d = discriminant_function(neuron_weights, pattern)
            if d < best_d:
                best_d = d
                best_neuron = (x, y)
    return best_neuron

def adapt_weights(winning_neuron, pattern, time):
    n = learning_rate(time)
    count = 0
    for y in range(LATTICE_HEIGHT):
        for x in range(LATTICE_WIDTH):
            T = neighbourhood_function(winning_neuron, (x,y), time)
            delta = pattern - WEIGHTS[y][x]
            update_delta = n * T * delta
            #print("WEIGHTS[y][x]", WEIGHTS[y][x])
            #print("n", n)
            #print("T", T)
            #print("delta", delta)
            #print("update_delta", update_delta)
            WEIGHTS[y][x] += update_delta
            #print("WEIGHTS[y][x]", WEIGHTS[y][x])
            #print()

def output_jpeg():
    pixel_list = []
    for y in range(LATTICE_HEIGHT):
        for x in range(LATTICE_WIDTH):
            pixel_list.append(tuple(map(int, WEIGHTS[y][x])))
    #print(pixel_list)


    # Create new image where colours are placed
    output = Image.new("RGB", (LATTICE_WIDTH, LATTICE_HEIGHT))
    output.putdata(pixel_list)
    output.save("som_colors_output.jpg", "JPEG")
    output.show()

def output_svg():
    output = svgwrite.Drawing(filename="som_colors_output.svg")
    cm = svgwrite.cm
    output_width = 30 
    r = output_width / float(LATTICE_WIDTH)
    for y in range(LATTICE_HEIGHT):
        for x in range(LATTICE_WIDTH):
            color = tuple(map(int, WEIGHTS[y][x]))
            output.add(svgwrite.shapes.Rect((x*r*cm,y*r*cm), (r*cm,r*cm), fill=svgwrite.rgb(*color)))
    output.save()


if __name__ == "__main__":
#if 0:
    if len(sys.argv) < 2:
        print("Please give path to image file as argument")
    else:
        image_path = sys.argv[1]
        im = Image.open(image_path)
        print("Opening image: " + image_path)
        print(im.size)

        # List of all the pixel colour values
        im_data = list(im.getdata())
        random.shuffle(im_data)
        print("Number of pixels: ", len(im_data))

        # TRAINING STAGE
        # Iterate through all inputs in batches
        print("TRAINING")
        for time in range(NUM_EPOCHS):
            print("Epoch: " + str(time))
            for j in range(EPOCH_SIZE):
                pixel_index = (time * EPOCH_SIZE + j) % len(im_data)
                pixel = np.array(im_data[pixel_index])
                winning_neuron = find_winning_neuron(pixel)
                adapt_weights(winning_neuron, pixel, time)


        # OUTPUT
        """
        print("OUTPUT")
        pixel_grid = np.zeros((LATTICE_HEIGHT, LATTICE_WIDTH, 3))
        # Check where every input is placed in the SOM
        for i in range(len(im_data)):
            pixel = np.array(im_data[i])
            (x, y) = find_winning_neuron(pixel)
            pixel_grid[y][x] = pixel
            print(i, pixel, (x,y))
        """
        output_svg()
