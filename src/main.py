from os import listdir
from os.path import isfile, join
from sklearn import preprocessing
import numpy as np
import search
import pandas as pd
import argparse
import sys
import pickle

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
    data = np.array(data)
    n, d = data.shape

    averageVector = data[0]
    for i in range(1, len(data)):
        averageVector += data[i]
    averageVector = averageVector / len(averageVector)

    print("Substracting average vector to all vectors")
    for i in range(0, len(data)):
        data[i] = data[i] - averageVector

    print("Computing covMat of DT")
    covMat = np.cov(data.T, rowvar=False)
    print(len(covMat))
    print(len(covMat[0]))

    print("Computing eigen(vector|values)")
    eigenValues, eigenVectors = np.linalg.eig(covMat)
    print(len(eigenVectors))
    print(len(eigenValues))

    eigenFaces = data.T.dot(eigenVectors)
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
    print("Transforming gallery")
    model["gallery"] = transformDataset(data, model["eigenFaces"], model["averageVector"])

    print("Saving")
    saveModel(model)

def saveModel(m):
    with open("model.pkl", "wb") as f:
        pickle.dump(m, f, pickle.HIGHEST_PROTOCOL)

def loadModel(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def transformGalleryAndSave(path, eigenFaces,):
    print("Loading gallery")
    gallery = loadImageToArray(path)
    print("Transforming gallery")
    gallery = transformDataset(gallery, eigenFaces, averageVector)

def loadAndTransform(path, m):
    data = loadImageToArray(path)
    return transformDataset(data, m['eigenFaces'], m['averageVector'])

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--action", type=str,
                    help="[generateModel, findBestR, evaluateRadius]")

args = parser.parse_args()
if args.action == "generateModel":
    print("Generating model")
    trainModelAndSave(DATASET_DIR_1)

elif args.action == "findBestR":
    print("Loading model")
    m = loadModel("model.pkl")

    print("Loading and transform posProbes")
    posProbes = loadAndTransform(DATASET_DIR_POSITIVE, m)

    print("Computing bestR")
    bestR = findBestR(m["gallery"], posProbes)
    print("Best R is:", bestR)

    print("Updating model")
    m["r"] = bestR
    saveModel(m)

elif args.action == "evaluateRadius":
    print("Loading model")
    m = loadModel("model.pkl")
    if not "r" in m:
        print("You should compute bestR before using the model")
        sys.exit(1)

    print("Loading and transform posProbes")
    posProbes = loadAndTransform(DATASET_DIR_POSITIVE, m)

    print("Loading and transform negProves")
    negProbes = loadAndTransform(DATASET_DIR_NEGATIVE, m)

    print("Evaluating radius")
    evaluateRadius(m["gallery"], posProbes, negProbes, m["r"])
else:
    parser.print_help()
    sys.exit(1)

sys.exit(0)
