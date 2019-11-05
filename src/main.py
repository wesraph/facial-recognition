from os import listdir
from os.path import isfile, join
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

print("Loading posProbes")
posProbes = loadImageToArray(DATASET_DIR_POSITIVE)

print("Loading negProbes")
negProbes = loadImageToArray(DATASET_DIR_NEGATIVE)

print("Loading dataset")
gallery = loadImageToArray(DATASET_DIR_1)

print("Querying pos")
indices, posProbesResult = search.radius_search(gallery, posProbes[0], r=100)

print("Indices" , indices)
print(posProbesResult)
print(len(posProbesResult))
