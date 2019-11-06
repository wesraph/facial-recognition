from os import listdir
from os.path import isfile, join
# from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import matplotlib.image as mpimg
import numpy as np
import functools
import search
import pandas as pd

# Path of the dataset 1
# DATASET_DIR_1="../data/dataset1/images/"
DATASET_DIR_1="../data/dataset1/images/"
DATASET_DIR_POSITIVE="../data/dataset1/positive/"
DATASET_DIR_NEGATIVE="../data/dataset1/negative/"

def loadImageToArray(path):
    jpg_list = [f for f in listdir(path) if isfile(join(path, f))]

    if(len(jpg_list) == 0):
        print("No files found at " + path)
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

def transformDataset(data, eigenFaces, averageVector):
    data = np.array(data)
    return np.subtract(data, averageVector).dot(eigenFaces)
    # return np.array(
            # list(
                # map(lambda q: np.array(q - averageVector).T.dot(eigenFaces), data
            # )))


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

def applyPCA(data):
    print("Compute average vector")
    n, d = np.array(data).shape

    averageVector = data[0]
    for i in range(1, len(data)):
        averageVector += data[i]
    averageVector = averageVector / len(averageVector)

    print("Substracting average vector to all vectors")
    for i in range(0, len(data)):
        data[i] = data[i] - averageVector

    print("Computing covMat of DT")
    covMat = np.cov(np.array(data).T, rowvar=False)
    print(len(covMat))
    print(len(covMat[0]))

    print("Computing eigen(vector|values)")
    eigenValues, eigenVectors = np.linalg.eig(covMat)
    print(len(eigenVectors))
    print(len(eigenValues))

    eigenFaces = np.array(data).T.dot(eigenVectors)
    eigenFaces = preprocessing.normalize(eigenFaces)

    eigenValues = eigenValues * ((d - 1) / (n - 1))

    return eigenFaces, eigenValues, averageVector

def importDataset(path):
    data = np.load(path).tolist()
    for i in range(0, len(data)):
        data[i] = np.array(data[i])

    return data

def toListNDArray(data):
    dlist = data.tolist()
    for i in range(0, len(dlist)):
        data[i] = np.array(data[i])
    return data

def trainModelAndSave(path):
    print("Loading dataset (images to array)")
    data = loadImageToArray(path)

    model = {}
    model["eigenFaces"], model["eigenValues"], model["averageVector"] = applyPCA(data)

    print("Reloading data")
    data = []
    data = loadImageToArray(path)
    model["gallery"] = transformDataset(data, eigenFaces, averageVector)

    print("Saving")
    np.save("model.npy", model)

def loadModel(path):
    model = np.load(path)
    return model["eigenFaces"], model["eigenValues"], model["averageVector"]

def transformGalleryAndSave(path, eigenFaces,):
    print("Loading gallery")
    gallery = loadImageToArray(path)
    print("Transforming gallery")
    gallery = transformDataset(gallery, eigenFaces, averageVector)


trainModelAndSave(DATASET_DIR_1)

sys.exit(0)

eigenFaces, eigenValues, averageVector = loadModel("model.npy")

print("Loading gallery")
gallery = loadImageToArray(DATASET_DIR_1)
print("Transforming gallery")
gallery = transformDataset(gallery, eigenFaces, averageVector)

print("Loading posProbes")
posProbes = loadImageToArray(DATASET_DIR_POSITIVE)
print("Transforming posProbes")
posProbes = transformDataset(posProbes, eigenFaces, averageVector)

print("Loading negProbes")
negProbes = loadImageToArray(DATASET_DIR_NEGATIVE)
print("Transforming negProbes")
negProbes = transformDataset(negProbes, eigenFaces, averageVector)

print("Computing bestR")
bestR = findBestR(gallery, posProbes)
print(bestR)

print("Evaluating radius")
evaluateRadius(gallery, posProbes, negProbes, bestR)

# print("Loading negProbes")
# negProbes = loadImageToArray(DATASET_DIR_NEGATIVE)


# print("Loading dataset gallery")
# gallery = np.array(loadImageToArray(DATASET_DIR_1))

# print("Reducing gallery")
# print(type(gallery))
# eigenFaces = gallery.T.dot(eigenVectors)

# print("Loading posProbes")
# posProbes = loadImageToArray(DATASET_DIR_POSITIVE)
# redPosProbes = applyPCA3(posProbes)

# sys.exit(0)

# print("Loading negProbes")
# negProbes = loadImageToArray(DATASET_DIR_NEGATIVE)
# redNegProbes = applyPCA(negProbes)

# bestR = findBestR(redGallery, redPosProbes)
# bestR = 2027817.0
# print("Best R:", bestR)

# Evaluate radius
# evaluateRadius(gallery, posProbes, negProbes, bestR)


# print("Indice ", findMinR(gallery, negProbes))
