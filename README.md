# Facial recognition
The goal of this tool is to implemente the algorithm of EigenFaces.

# EigenFaces
EigenFaces is the name given to a set of eigen vectors when they are used in computer vison problem. The goal here is to reduce the number of stored components to improve data size and computing speed

# Installation
This tool depends on numpy¸ matplotlib, pandas and sklearn

# Dataset generation
In order to use this tool, you need to download a dataset of faces centered (https://cswww.essex.ac.uk/mv/allfaces/)

# Usage
```
usage: main.py [-h] [-gr] [-gp] [-gm] [-sm] [-s] [-gie] [-b] [-bcn] [-br]
               [-cm] [-m MODEL] [-ma MODELA] [-mb MODELB] [-r RADIUS]
               [-n NCOMPONENTS]

optional arguments:
  -h, --help            show this help message and exit
  -gr, --generateRawModel
                        generate a model without transformations
  -gp, --generatePCAModel
                        generate a base model (PCA with all principal
                        components)
  -gm, --generateReducedModel
                        generate a reduced model from the base model
  -sm, --setModelSettings
                        set a setting of the model
  -s, --showModelSettings
                        plot the eigenValues of the model
  -gie, --generateImageEigenFaces
                        generate the eigenfaces (image) and save it
  -b, --benchmark       evalute the accuracy of a model
  -bcn, --benchmarkCompsNb
                        benchmark the impact of the number of kept principal
                        components of the model
  -br, --benchmarkRadius
                        benchmark the impact of R values
  -cm, --compareModels  compare the performances of two models
  -m MODEL, --model MODEL
                        path of the model to load
  -ma MODELA, --modelA MODELA
                        path of the model A to load
  -mb MODELB, --modelB MODELB
                        path of the model B to load
  -r RADIUS, --radius RADIUS
                        the r value to set
  -n NCOMPONENTS, --nComponents NCOMPONENTS
                        the number of components to use
```

# Generated faces
![gallery of eigen faces](https://raw.githubusercontent.com/wesraph/facial-recognition/master/img/gallery.png)

![gallery of summed eigen faces](https://raw.githubusercontent.com/wesraph/facial-recognition/master/img/gallerySummed.png)
