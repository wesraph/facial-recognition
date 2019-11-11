import numpy as np
import shared


def showModel(path):
    print("Loading model")
    m = shared.loadModel(path)

    print("R:", m["r"])
    print("Number of components:", len(m["gallery"][0]))

    if not "eigenValues" in m:
        return

    print(m["originalSumEigenValues"])
    ev = np.flip(np.sort(m["eigenValues"]))
    ev = ev / m["originalSumEigenValues"]
    shared.plt.plot(ev[:100])

    inertia = round(np.sum(ev[:len(m["gallery"][0])]) * 100, 3)
    print("Inertia:", inertia, "%")

    shared.plt.show()

def setModelSettings(modelPath, nComponents, r):
    m = shared.loadModel(modelPath)

    if nComponents:
        m = shared.generator.reduceModel(nComponents, args.model, r=False)
    if r:
        m["r"] = r

    shared.saveModel(m, modelPath)
