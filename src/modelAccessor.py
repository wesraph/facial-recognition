import numpy as np
import shared
import math
from PIL import Image

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

def generateEigenFacesPictures(path, number=10):
    print("Showing eigenFaces")

    print("Loading model")
    m = shared.loadModel(path)

    m["eigenFaces"] = m["eigenFaces"].T
    w, h = int(math.sqrt(len(m["eigenFaces"][0]))), int(math.sqrt(len(m["eigenFaces"][0])))

    gallery = np.zeros((h * number, w * number), dtype=np.uint8)

    summed = np.zeros((h, w), dtype=np.uint8)
    gallerySummed = np.zeros((h * number, w * number), dtype=np.uint8)

    for i in range(1, (number * number ) + 1):
        print(i)
        data = np.zeros((h, w), dtype=np.uint8)
        r = len(m["eigenFaces"]) - i
        vmax = np.amax(m["eigenFaces"][r])
        im = np.split(m["eigenFaces"][r], w)
        for ih in range(0, h):
            for iw in range(0, w):
                gallery[iw + w * int((i - 1) / number)][ih + h * int((i - 1) % number)] = im[iw][ih] / vmax * 255
                gallerySummed[iw + w * int((i - 1) / number)][ih + h * int((i - 1) % number)] = summed[iw][ih] + im[iw][ih] / vmax * 255

    print("Gallery of eigenFaces saved to gallery.png")
    img = Image.fromarray(gallery)
    img.save("gallery.png")

    print("Gallery of summed eigenFaces saved to gallerySummed.png")
    img = Image.fromarray(gallerySummed)
    img.save("gallerySummed.png")

