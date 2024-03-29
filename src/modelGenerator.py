from sklearn import preprocessing
import numpy as np
import shared
import random

def findBestR(m, isRandom=True, limit=100):
    print("Computing average R")
    gallery = m["gallery"]
    lenGallery = len(m["gallery"])
    iterator = []

    if isRandom:
       iterator = [random.randint(0, lenGallery-1) for i in range(0, limit)]
    else:
        iterator = range(0, lenGallery)

    lenIterator = len(iterator)
    u = 0
    dmax = []

    for i in iterator:
        print(round((u / lenIterator * 100), 3), "%")
        u = u + 1
        sliced = np.concatenate((gallery[:i],gallery[i+1:]), axis=0)
        distances = shared.search.compute_distances(sliced, gallery[i])
        min = np.amin(distances)
        dmax.append(min)

    dmax = np.array(dmax)
    mean = np.mean(dmax)
    standDeviation = np.sqrt(np.var(dmax))
    sorted =  dmax[np.where(dmax <= mean + standDeviation)]
    bestR = np.amax(sorted)
    return bestR

def generateRawModel(path):
    print("Creating raw model (no optimisation)")
    m  = {}
    m["gallery"] = shared.loadImageToArray(shared.DATASET_DIR_1)
    m["r"] = findBestR(m, limit=200)
    shared.saveModel(m, path)

def generatePCAModel(path):
    print("Creating pca model")
    print("Loading dataset (images to array)")
    data = shared.loadImageToArray(shared.DATASET_DIR_1)

    model = {}
    model["eigenFaces"], model["eigenValues"], model["averageVector"] = applyPCA(data)

    print("Transforming gallery")
    model["gallery"] = shared.transformDataset(data, model["eigenFaces"], model["averageVector"])
    model["originalSumEigenValues"] = np.sum(model["eigenValues"])

    print("Searching bestR")
    model["r"] = findBestR(model, limit=200)

    shared.saveModel(model, path)

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

def reduceSpace(m, eigenValues, nComps):
    bestEV = np.flip(np.argsort(eigenValues))[:nComps]

    if type(m) is dict:
        redGallery = np.take(m["gallery"], bestEV, axis=1)
        redEigenFaces = np.take(m["eigenFaces"], bestEV, axis=1)
        redEigenValues = np.take(m["eigenValues"], bestEV)
        return redGallery, redEigenFaces, redEigenValues
    else:
        redGallery = np.take(m, bestEV, axis=1)
        return redGallery

def generateReducedModel(modelB, nComponents, r=True):
    mb = reduceModel(nComponents, r=True)
    shared.saveModel(mb, modelB)

def reduceModel(nComponents, path, r=True):

    ma = shared.loadModel(path)

    if(nComponents > len(ma["gallery"][0])):
        print("ERROR: Can't have more components than generated by the original PCA")
        return

    print("Reducing to", nComponents, "components")
    mb = {}
    mb["gallery"], mb["eigenFaces"], mb["eigenValues"] = reduceSpace(ma, ma["eigenValues"], nComponents)
    mb["originalSumEigenValues"] = ma["originalSumEigenValues"]
    mb["averageVector"] = ma["averageVector"]

    if r:
        print("Computing R")
        mb["r"] = findBestR(mb, limit=200)
    else:
        mb["r"] = ma["r"]

    return mb
