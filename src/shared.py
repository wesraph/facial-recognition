import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import re
import os

import modelGenerator as generator
import modelAccessor as accessor
import modelBenchmark as benchmark
import search


DATASET_DIR_1="../data/dataset1/images/"
DATASET_DIR_POSITIVE="../data/dataset1/positive/"
DATASET_DIR_NEGATIVE="../data/dataset1/negative/"

def saveModel(m, filename):
    if not os.path.isdir('./models') :
        os.mkdir('./models')
    with open("./models/" + filename + ".pkl", "wb") as f:
        pickle.dump(m, f, pickle.HIGHEST_PROTOCOL)
    print("Saved model", filename)

def loadModel(path):
    with open("./models/" + path + ".pkl", "rb") as f:
        return pickle.load(f)

def loadAndTransform(path, m):
    data = loadImageToArray(path)
    return transformDataset(data, m['eigenFaces'], m['averageVector'])

def loadImageToArray(path):
    jpg_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    if(len(jpg_list) == 0):
        print("No files found at " + path)
        sys.exit(1)

    print("Number of images: ", len(jpg_list))

    jpg_count = len(jpg_list)
    img_array = []
    for i in range(0, jpg_count):
        if not re.search("\.jpg$|\.png$", jpg_list[i]):
            continue
        img = mpimg.imread(path + jpg_list[i])
        img_array.append(np.array(img).flatten())

    return img_array


def transformDataset(data, eigenFaces, averageVector):
    data = np.array(data)
    return np.subtract(data, averageVector).dot(eigenFaces)