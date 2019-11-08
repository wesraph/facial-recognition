from os import listdir
from os.path import isfile, join
from sklearn import preprocessing
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
import search
import argparse
import sys
import pickle

DATASET_DIR_1="../data/dataset1/images/"
DATASET_DIR_POSITIVE="../data/dataset1/positive/"
DATASET_DIR_NEGATIVE="../data/dataset1/negative/"

def loadImageToArray(path):
    jpg_list = [f for f in listdir(path) if isfile(join(path, f))]

    if os.name == 'nt':
        if jpg_list[0] == "desktop.ini":
            jpg_list.pop(0)

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
    print("lenGallery", len(m["gallery"]))

    gallery = m["gallery"]
    lenGallery =len(m["gallery"][0])
    #origin = np.zeros((1, lenGallery))
    #distances = search.compute_distances(gallery, origin)
    #averageR = np.sum(distances) / len(distances)
    #print("Average R:", averageR/1000000000)
    r = 0
    iterator = []

    if isRandom:
       iterator = [random.randint(0, lenGallery) for i in range(0, limit)]
    else:
        iterator = range(0, lenGallery)

    lenIterator = len(iterator)

    u = 0

    dmax = []

    for i in iterator:
        print((u / lenIterator * 100))
        u = u + 1
        sliced = np.concatenate((gallery[:i],gallery[i+1:]), axis=0)
        distances = search.compute_distances(sliced, gallery[i])
        min = np.amin(distances)
        dmax.append(min)

    dmax = np.array(dmax)
    average = np.mean(dmax)
    sorted =  dmax[np.where(dmax < 2*average)]
    bestR = np.amax(sorted)
    return bestR

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
    print("False refused probes: ", falseRefusedProbes / (lenPosProbes + lenNegProbes))

    globalEfficiency = (acceptedProbes / (lenPosProbes + lenNegProbes)) + (refusedProbes / (lenPosProbes + lenNegProbes))
    return globalEfficiency

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
    eigenVectors = data.T.dot(eigenVectors)
    eigenFaces = preprocessing.normalize(eigenVectors)

    eigenValues = eigenValues * ((d - 1) / (n - 1))

    return eigenFaces, eigenValues, averageVector


def reduceSpaces(gallery, eigenValues, n=10):
    indices = np.flip(np.argsort(eigenValues))

    print(indices)

    newGallery = []
    for i in range(0, len(gallery)):
        el = []
        for u in range(0, n):
            el.append(gallery[i][indices[u]])
        newGallery.append(np.array(el))

    return np.array(newGallery)

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

def loadAndTransform(path, m):
    data = loadImageToArray(path)
    return transformDataset(data, m['eigenFaces'], m['averageVector'])

def settingsImpact(m):
    print("Loading and transform posProbes")
    posProbes = loadAndTransform(DATASET_DIR_POSITIVE, m)
    print("Loading and transform negProves")
    negProbes = loadAndTransform(DATASET_DIR_NEGATIVE, m)

    m["gallery"] = reduceSpaces(m["gallery"], m["eigenValues"])
    posProbes = reduceSpaces(posProbes, m["eigenValues"])
    negProbes = reduceSpaces(negProbes, m["eigenValues"])

    print(len(m["gallery"]))
    print("reduced gallery length", len(m["gallery"][0]))
    print("reduced posProbes length", len(posProbes[0]))
    print("reduced negProbes length", len(negProbes[0]))

    r = findBestR(m, isRandom=True, limit=200)
    print(r)

    percentageRange = np.arange(0, 0.5, 0.05)
    efficiencyAxis = []
    for percent in percentageRange:
        print( r + r*percent)
        efficiency = evaluateRadius(m["gallery"], posProbes, negProbes, r + r*percent)
        print(efficiency)
        efficiencyAxis.append(efficiency)
        print(percent*100, "%")
    plt.plot(percentageRange*100, efficiencyAxis)

# def reduceSpaces(m, posProbes, negProbes, nbEF = 20):
        # return [i[:nbEF] for i in m["gallery"]],[i[:nbEF] for i in posProbes],[i[:nbEF] for i in negProbes]

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--action", type=str,
                    help="[generateModel, findBestR, evaluateRadius]")

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

elif args.action == "settingsImpact":
    m = loadModel("model.pkl")
    settingsImpact(m)

elif args.action == "generateModel":
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
