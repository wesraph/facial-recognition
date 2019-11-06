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
    scaled = preprocessing.scale(data)
    scaled = scaled.T
    covMat = np.cov(scaled, rowvar=False)
    eigenValuesTrans, eigenVectorsTrans = np.linalg.eig(covMat)
    print(len(eigenVectorsTrans))
    print(len(eigenVectorsTrans[0]))

    eigenVectors = []
    for i in range(0, len(eigenVectorsTrans)):
        eigenVectors.append(scaled * eigenVectorsTrans[i])

    normalizedEVe = preprocessing.normalize(eigenVectors)
    # mainComps = data * normalizedEVe
    # return list(mainComps), eigenVectors, eigenValues
    eigenValues = eigenValuesTrans * ((len(data[0]) - 1) / (len(data) - 1))
    # return mainComps, normalizedEVe, eigenValues
    return normalizedEVe, eigenValues

def applyPCA3(data):
    print("Compute average vector")
    averageVector = data[0]
    for i in range(1, len(data)):
        averageVector += data[i]
    averageVector = averageVector / len(averageVector)

    print("Average vector is:", averageVector)

    print("Substracting average vector to all vectors")
    for i in range(0, len(data)):
        data[i] = data[i] - averageVector

    print("Computing covMat")
    covMat = np.cov(data, rowvar=False)

    print("Computing eigen(vector|values)")
    eigenValues, eigenVectors = np.linalg.eig(covMat)







def applyPCA2(data):
    print("Scaling data")
    print(len(data))
    print(len(data[0]))
    # TODO: Check
    scaled = preprocessing.scale(data).T
    print(len(scaled))
    print(len(scaled[0]))
    print("Transpose")
    print("Covariance")
    # TODO: check this
    covMat = np.cov(scaled, rowvar=False)
    print("Len covmat", len(covMat))
    print("Len covmat0", len(covMat[0]))

    print("Eigenvalues")
    eigenValuesTrans, eigenVectorsTrans = np.linalg.eig(covMat)
    print("lenEigenVectors", len(eigenVectorsTrans))
    print("lenEigenVectors0", len(eigenVectorsTrans[0]))

    eigenVectors = scaled * eigenVectorsTrans
    normalizedEVe = preprocessing.normalize(eigenVectors)
    print(eigenVectors)
    print(normalizedEVe)

    eigenValues = eigenValuesTrans * ((len(data[0]) - 1) / (len(data) - 1))

    mainComps = data * normalizedEVe
    print(mainComps)

    # return list(mainComps), eigenVectors, eigenValues
    return mainComps, normalizedEVe, eigenValues

# def transformQuery(query, eigenVectors):
    # return

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

print("Loading posProbes")
posProbes = loadImageToArray(DATASET_DIR_POSITIVE)
redPosProbes = applyPCA(posProbes)

print("Loading negProbes")
negProbes = loadImageToArray(DATASET_DIR_NEGATIVE)
redNegProbes = applyPCA(negProbes)

print("Loading dataset gallery")
gallery = loadImageToArray(DATASET_DIR_POSITIVE)
redGallery, eigenVectors, eigenValues = applyPCA(gallery)
print("Saving redGallery")
saveDataset("redgallery.npy", redGallery)
saveDataset("eigenVectors.npy", eigenVectors)
saveDataset("eigenValues.npy", eigenValues)
# redGallery = importDataset("PCA_dataset.npy")

print("object total: ", type(redGallery))
print("object total: ", type(gallery))
print("object total: ", type(redPosProbes))

print("element 0", type(redGallery[0]))
print("element 0", type(gallery[0]))
print("element 0", type(redPosProbes[0]))

print("dim", len(redGallery))
print("dim", len(gallery))
print("dim", len(redPosProbes))

print("dim 0", len(redGallery[0]))
print("dim 0", len(gallery[0]))
print("dim 0", len(redPosProbes[0]))

bestR = findBestR(redGallery, redPosProbes)
# bestR = 2027817.0
print("Best R:", bestR)

# Evaluate radius
# evaluateRadius(gallery, posProbes, negProbes, bestR)


# print("Indice ", findMinR(gallery, negProbes))
