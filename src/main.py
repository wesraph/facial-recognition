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

def transformDataset(data, eigenFaces):
    data = preprocessing.normalize(data)
    return np.array(
            list(
                map(lambda q: np.array(q).T.dot(eigenFaces), data
            )))


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
    averageVector = data[0]
    for i in range(1, len(data)):
        averageVector += data[i]
    averageVector = averageVector / len(averageVector)

    print("Substracting average vector to all vectors")
    for i in range(0, len(data)):
        data[i] = data[i] - averageVector
    # From here, data is centered

    print("Computing covMat of DT")
    covMat = np.cov(np.array(data).T, rowvar=False)
    print(len(covMat))
    print(len(covMat[0]))

    print("Computing eigen(vector|values)")
    eigenValues, eigenVectors = np.linalg.eig(covMat)
    print(len(eigenVectors))
    print(len(eigenValues))

    eigenFaces = np.array(data).T.dot(eigenVectors)
    eigenVectors = preprocessing.normalize(eigenVectors)

    return eigenVectors, eigenFaces, averageVector

def saveDataset(path, data):
    return np.save(path, data)

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

def trainAndSave(path):
    print("Loading dataset (images to array)")
    data = loadImageToArray(path)

    print("Computing eigenFaces")
    eigenVectors, eigenFaces, averageVector = applyPCA(data)

    print("Saving")
    saveDataset("eigenFaces.npy", eigenFaces)
    saveDataset("eigenVectors.npy", eigenVectors)
    saveDataset("averageVector.npy", averageVector)


def transformQuery(q, eigenFaces):
    return np.array(q).T.dot(eigenFaces)

# trainAndSave(DATASET_DIR_1)

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    # help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
                    # const=sum, default=max,
                    # help='sum the integers (default: find the max)')

trainAndSave(DATASET_DIR_1)
sys.exit(0)

print("Loading eigenVectors")
# eigenVectors = importDataset("./eigenVectors.npy")
eigenVectors = np.load("eigenVectors.npy")

print("Loading eigenFaces")
eigenFaces = np.load("eigenFaces.npy")

print(len(eigenFaces))
print(len(eigenFaces[0]))
print(len(eigenVectors))
print(len(eigenVectors[0]))

print("Loading posProbes")
posProbes = loadImageToArray(DATASET_DIR_POSITIVE)
print("Transforming posProbes")
posProbes = transformDataset(posProbes, eigenFaces)

print("Loading negProbes")
negProbes = loadImageToArray(DATASET_DIR_NEGATIVE)
print("Transforming negProbes")
negProbes = transformDataset(negProbes, eigenFaces)

print("Computing bestR")
# bestR = findBestR(eigenFaces, posProbes)
bestR = 2826968800000.0
print("bestR", bestR)

print("Evaluating radius")
evaluateRadius(eigenFaces, posProbes, negProbes, bestR)

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

# print("Loading dataset gallery")
# gallery = loadImageToArray(DATASET_DIR_POSITIVE)
# redGallery, eigenVectors, eigenValues = applyPCA(gallery)
# print("Saving redGallery")
# saveDataset("redgallery.npy", redGallery)
# saveDataset("eigenVectors.npy", eigenVectors)
# saveDataset("eigenValues.npy", eigenValues)
# redGallery = importDataset("PCA_dataset.npy")

# bestR = findBestR(redGallery, redPosProbes)
# bestR = 2027817.0
# print("Best R:", bestR)

# Evaluate radius
# evaluateRadius(gallery, posProbes, negProbes, bestR)


# print("Indice ", findMinR(gallery, negProbes))
