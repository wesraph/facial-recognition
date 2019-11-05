from os import listdir
from os.path import isfile, join
# from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import matplotlib.image as mpimg
import numpy as np
import functools
import search

# Path of the dataset 1
# DATASET_DIR_1="../data/dataset1/images/"
DATASET_DIR_1="../data/dataset1/images/"
DATASET_DIR_POSITIVE="../data/dataset1/positive/"
DATASET_DIR_NEGATIVE="../data/dataset1/negative/"

def loadImageToArray(path):
    jpg_list = [f for f in listdir(path) if isfile(join(path, f))]
    # print(onlyfiles)
    if(len(jpg_list) == 0):
        print("No file found at " + path)
        sys.exit(1)

    print("Number of images: ", len(jpg_list))

    jpg_count = len(jpg_list)
    img_array = []
    for i in range(0, jpg_count):
        img = mpimg.imread(path + jpg_list[i])
        img_array.append(np.array(img).flatten())

    return img_array

def findBestR(gallery, posProbes):
    r = 0
    lenPosProbes = len(posProbes)
    for i in range(0, lenPosProbes):
        distances = search.compute_distances(gallery, posProbes[i])
        dmax = np.amin(distances)
        if dmax > r:
            r = dmax
    return r

def evaluateRadius(gallery, posProbes, negProbes, r):
    lenPosProbes = len(posProbes)
    lenNegProbes = len(negProbes)

    acceptedProbes = 0
    falseAcceptedProbes = 0

    refusedProbes = 0
    falseRefusedProbes = 0

    for i in range(0, lenPosProbes):
        indices, results = search.radius_search(gallery, posProbes[i], r=r)
        if(len(indices) != 0):
            acceptedProbes += 1
        else:
            falseRefusedProbes += 1

    for i in range(0, lenNegProbes):
        indices, results = search.radius_search(gallery, negProbes[i], r=r)
        if(len(indices) != 0):
            falseAcceptedProbes += 1
        else:
            refusedProbes += 1

    print("Accepted probes: ", acceptedProbes / (lenPosProbes + lenNegProbes))
    print("False accepted probes: ", falseAcceptedProbes / (lenPosProbes + lenNegProbes))
    print("Refused probes: ", refusedProbes / (lenPosProbes + lenNegProbes))
    print("Accepted probes: ", falseRefusedProbes / (lenPosProbes + lenNegProbes))

print("Loading posProbes")
posProbes = loadImageToArray(DATASET_DIR_POSITIVE)

print("Loading negProbes")
negProbes = loadImageToArray(DATASET_DIR_NEGATIVE)

print("Loading dataset")
gallery = loadImageToArray(DATASET_DIR_1)

# bestR = findBestR(gallery, posProbes)
bestR = 2027817.0
print("Best R:", bestR)

evaluateRadius(gallery, posProbes, negProbes, bestR)

# print("Indice ", findMinR(gallery, negProbes))

print("Querying pos")

# indices, posProbesResult = search.radius_search(gallery, negProbes[0], r=bestR)

print(posProbesResult)
print(indices)
print(len(indices))
