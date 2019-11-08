from os import listdir
from os.path import isfile, join
from sklearn import preprocessing
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
import search
import argparse
import sys
import pickle
import time

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

def findBestR(model, isRandom=False, limit=100):
    print("Computing average R")
    gallery = m["gallery"]
    averageVector = np.mean(gallery, axis=0)
    distances = search.compute_distances(gallery, averageVector)
    averageR = np.sum(distances) / len(distances)

    r = 0
    lenGallery = len(gallery)
    iterator = []

    if isRandom:
       iterator = [random.randint(0, lenGallery) for i in range(0, limit)]
    else:
        iterator = range(0, lenGallery)

    lenIterator = len(iterator)

    u = 0
    for i in iterator:
        print((u / lenIterator * 100))
        u = u + 1

        sliced = np.concatenate([gallery[:i],gallery[i+1:]])

        distances = search.compute_distances(sliced, gallery[i])
        dmax = np.amin(distances)
        if dmax > r and dmax < 2 * averageR:
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

    duration = time.time()
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

    duration = duration - time.time()
    print("Accepted probes: ", acceptedProbes / (lenPosProbes + lenNegProbes))
    print("False accepted probes: ", falseAcceptedProbes / (lenPosProbes + lenNegProbes))
    print("Refused probes: ", refusedProbes / (lenPosProbes + lenNegProbes))
    print("Accepted probes: ", falseRefusedProbes / (lenPosProbes + lenNegProbes))

    results = {}
    results["accuracy"] = (acceptedProbes + refusedProbes) / (acceptedProbes + falseRefusedProbes + refusedProbes + falseRefusedProbes)
    results["precision"] = (acceptedProbes) / (acceptedProbes + falseAcceptedProbes)
    results["sensibility"] = (acceptedProbes) / (acceptedProbes + falseRefusedProbes)
    results["specificity"] = (refusedProbes) / (refusedProbes + falseAcceptedProbes)
    results["duration"] = duration

    return results


def applyPCA(data):
    print("Compute average vector")
    data = np.array(data)
    n, d = data.shape

    print("Centering matrix")
    averageVector = np.mean(data, axis=0)
    data = np.subtract(data, averageVector)

    print("Computing covMat of DT")
    covMat = np.cov(data.T, rowvar=False)

    print("Computing eigen(vector|values)")
    eigenValues, eigenVectors = np.linalg.eigh(covMat)
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
    saveModel(model, "model.pkl")

def saveModel(m, filename):
    with open(filename, "wb") as f:
        pickle.dump(m, f, pickle.HIGHEST_PROTOCOL)

def loadModel(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def transformGalleryAndSave(path, eigenFaces, averageVector):
    print("Loading gallery")
    gallery = loadImageToArray(path)
    print("Transforming gallery")
    gallery = transformDataset(gallery, eigenFaces, averageVector)

def loadAndTransform(path, m):
    data = loadImageToArray(path)
    return transformDataset(data, m['eigenFaces'], m['averageVector'])

def perfCompare():
    print("Comparing performances")

    print("Loading optimised model")
    m = loadModel("model.pkl")
    print("Loading posProbes")
    posProbes = loadImageToArray(DATASET_DIR_POSITIVE)
    negProbes = loadImageToArray(DATASET_DIR_NEGATIVE)

    print("Transforming posProbes")
    transformTime = time.time()
    posProbes = transformDataset(posProbes, m["eigenFaces"], m["averageVector"])
    negProbes = transformDataset(negProbes, m["eigenFaces"], m["averageVector"])
    transformTime = time.time() - transformTime


    results = {}
    print("Mesuring perfomances of optimised model")
    results["m"] = evaluateRadius(m["gallery"], posProbes, negProbes, m["r"])
    results["m"]["duration"] += transformTime

    print("Loading raw model")
    m = loadModel("rawModel.pkl")
    print("Reloading probes")
    posProbes = loadImageToArray(DATASET_DIR_POSITIVE)
    negProbes = loadImageToArray(DATASET_DIR_NEGATIVE)

    print("Mesuring performances of raw model")
    results["rmm"] = evaluateRadius(m["gallery"], posProbes, negProbes, m["r"])

    print(results)

def queryModel(m, query):
    indices, results = search.radius_search(m["gallery"], query, r=m["r"])
    minDist = np.amin(indices)
    return  minDist < m["r"]

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--action", type=str,
                    help="[plotEigenValues, generateModel, findBestR, createRawModel, pgenerateModel, findBestR, evaluateRadius, plotEigenValues, createRawModel,erfCompare, evaluateRadius]")



args = parser.parse_args()
if args.action == "plotEigenValues":
    print("Printing eigenValues")

    m = loadModel("model.pkl")
    ev = np.flip(np.sort(m["eigenValues"]))
    average = np.sum(ev)
    ev = ev / average
    plt.plot(ev[:100])
    print(ev)
    plt.show()

if args.action == "generateModel":
    print("Generating model")
    trainModelAndSave(DATASET_DIR_1)

elif args.action == "findBestR":
    print("Loading model")
    m = loadModel("model.pkl")

    print("Computing bestR")
    bestR = findBestR(m, isRandom=True, limit=200)
    print("Best R is:", bestR)

    print("Updating model")
    m["r"] = bestR
    saveModel(m, "model.pkl")

elif args.action == "generateRawModel":
    print("Creating raw model (no optimisation)")
    m  = {}
    m["gallery"] = loadImageToArray(DATASET_DIR_1)
    m["r"] = findBestR(m, isRandom=True, limit=200)

    print("Saving model")
    saveModel(m, "rawModel.pkl")

elif args.action == "perfCompare":
    perfCompare()

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
