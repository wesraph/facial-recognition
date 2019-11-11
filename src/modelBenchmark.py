import numpy as np
import shared
import time


def evaluateRadius(gallery, posProbes, negProbes, r):
    lenPosProbes = len(posProbes)
    lenNegProbes = len(negProbes)

    acceptedProbes = 0
    falseAcceptedProbes = 0

    refusedProbes = 0
    falseRefusedProbes = 0

    duration = time.time()
    for i in range(0, lenPosProbes):
        indices, results = shared.search.radius_search(gallery, posProbes[i], r=r)
        if(len(indices) != 0):
            acceptedProbes += 1
        else:
            falseRefusedProbes += 1

    for i in range(0, lenNegProbes):
        indices, results = shared.search.radius_search(gallery, negProbes[i], r=r)
        if(len(indices) != 0):
            falseAcceptedProbes += 1
        else:
            refusedProbes += 1

    duration = time.time() - duration

    results = {}
    results["accuracy"] = (acceptedProbes + refusedProbes) / (acceptedProbes + falseRefusedProbes + refusedProbes + falseAcceptedProbes)
    results["precision"] = 0 if acceptedProbes + falseAcceptedProbes == 0 else (acceptedProbes) / (acceptedProbes + falseAcceptedProbes)
    results["sensibility"] = 0 if acceptedProbes + falseRefusedProbes == 0 else (acceptedProbes) / (acceptedProbes + falseRefusedProbes)
    results["specificity"] = 0 if refusedProbes + falseAcceptedProbes == 0 else (refusedProbes) / (refusedProbes + falseAcceptedProbes)
    results["duration"] = duration

    return results

def benchmark(modelPath):
    m = shared.loadModel("pcaModel")
    if not "r" in m:
        print("You should compute bestR before using the model")
        shared.sys.exit(1)

    printPerfResults(getPerfomances(modelPath))
    

def benchmarkCompsNb(modelPath):
    print("Benchmarking number of components")
    m = shared.loadModel(modelPath)

    if not "eigenValues" in m:
        print("You cannot benchmark the number of components on a raw model")
        shared.sys.exit(1)

    print("Loading and transform posProbes")
    posProbes = shared.loadAndTransform(shared.DATASET_DIR_POSITIVE, m)
    print("Loading and transform negProves")
    negProbes = shared.loadAndTransform(shared.DATASET_DIR_NEGATIVE, m)

    # TODO: Improve range (maybe 2 benchmarks ?)
    nPropsRange = np.arange(1, 150, 10)
    accuracyAxis = []
    speedAxis = []
    for nProps in nPropsRange:
        redPosProbes = shared.generator.reduceSpace(posProbes, m["eigenValues"], nProps)
        redNegProbes = shared.generator.reduceSpace(negProbes, m["eigenValues"], nProps)
        gallery = shared.generator.reduceSpace(m["gallery"], m["eigenValues"], nProps)
        results = evaluateRadius(gallery, redPosProbes, redNegProbes, m["r"])

        printPerfResults(results)
        print("Number of components:", nProps)

        accuracyAxis.append(results["accuracy"])
        speedAxis.append(results["duration"])

    shared.plt.plot(nPropsRange, speedAxis)
    shared.plt.show()
    shared.plt.plot(nPropsRange, accuracyAxis)


def benchmarkR(modelPath):
    print("Benchmarking the radius")
    m = shared.loadModel(modelPath)

    if "eigenValues" in m:
        print("Loading and transform posProbes")
        posProbes = shared.loadAndTransform(shared.DATASET_DIR_POSITIVE, m)
        print("Loading and transform negProves")
        negProbes = shared.loadAndTransform(shared.DATASET_DIR_NEGATIVE, m)
    else:
        posProbes = shared.loadImageToArray(shared.DATASET_DIR_POSITIVE)
        negProbes = shared.loadImageToArray(shared.DATASET_DIR_NEGATIVE)

    percentageRange = np.round(np.arange(-1, 2, 0.25), 2)    
    accuracyAxis = []
    sensibilityAxis = []
    specificity = []
    r = m["r"]
    for percent in percentageRange:
        results = evaluateRadius(m["gallery"], posProbes, negProbes, r + r*percent)
        accuracyAxis.append(results["accuracy"])
        sensibilityAxis.append(results["sensibility"])
        specificity.append(results["specificity"])
        printPerfResults(results)
        print("Result for difference of ",percent*100, "%")

    shared.plt.plot([str(nb*100)+"%" for nb in percentageRange], accuracyAxis, label="Accuracy")
    shared.plt.plot([str(nb*100)+"%" for nb in percentageRange], sensibilityAxis, label="Senibility")
    shared.plt.plot([str(nb*100)+"%" for nb in percentageRange], specificity, label="Specificity")
    shared.plt.plot()
    shared.plt.legend()
    shared.plt.show()

def getPerfomances(modelPath):
    print("Loading ", modelPath)
    m = shared.loadModel(modelPath)

    print("Loading probes")
    posProbes = shared.loadImageToArray(shared.DATASET_DIR_POSITIVE)
    negProbes = shared.loadImageToArray(shared.DATASET_DIR_NEGATIVE)

    transformTime = time.time()
    if "eigenFaces" in m:
        print("Transforming probes")
        posProbes = shared.transformDataset(posProbes, m["eigenFaces"], m["averageVector"])
        negProbes = shared.transformDataset(negProbes, m["eigenFaces"], m["averageVector"])
    transformTime = time.time() - transformTime

    print("Mesuring perfomances of ", modelPath)
    results = evaluateRadius(m["gallery"], posProbes, negProbes, m["r"])
    results["duration"] += transformTime

    return results

def printPerfResults(perf):
    print(" Accuracy:", round(perf["accuracy"], 3))
    print(" Precision:",  round(perf["precision"], 3))
    print(" Sensibility:",round(perf["sensibility"], 3))
    print(" Specificity:",round(perf["specificity"], 3))
    print(" Duration:",   round(perf["duration"], 3))


def perfCompare(modelAPath, modelBPath):
    print("Comparing performances")
    maResults = getPerfomances(modelAPath)
    mbResults = getPerfomances(modelBPath)

    print("Model A:")
    printPerfResults(maResults)

    print("Model B:")
    printPerfResults(mbResults)

    print("Acceleration factor (b / a):", round(mbResults["duration"] / maResults["duration"] , 3))