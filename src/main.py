import shared
import argparse

parser = argparse.ArgumentParser()

#######################################    FUNCTIONS    ######################################################

#Generating a model
parser.add_argument("-gr", "--generateRawModel", action="store_true", help="generate a model without transformations")
parser.add_argument("-gp", "--generatePCAModel", action="store_true", help="generate a base model (PCA with all principal components)")
parser.add_argument("-gm", "--generateReducedModel", action="store_true", help="generate a reduced model from the base model")

#See & Modify a model
parser.add_argument("-sm", "--setModelSettings", action="store_true", help="set a setting of the model")
parser.add_argument("-s", "--showModelSettings", action="store_true", help="plot the eigenValues of the model")

#Benchmark a model
parser.add_argument("-b", "--benchmark", action="store_true", help="evalute the accuracy of a model")
parser.add_argument("-bcn", "--benchmarkCompsNb", action="store_true", help="benchmark the impact of the number of kept principal components of the model")
parser.add_argument("-br", "--benchmarkRadius", action="store_true", help="benchmark the impact of R values")

#Compare two models
parser.add_argument("-cm", "--compareModels", action="store_true", help="compare the performances of two models")

#######################################    ARGUMENTS     ######################################################

parser.add_argument("-m", "--model", default="./models/pcaModel.pkl", help="path of the model to load")
parser.add_argument("-ma", "--modelA", default="./models/pcaModel.pkl", help="path of the model A to load")
parser.add_argument("-mb", "--modelB", help="path of the model B to load")
parser.add_argument("-r", "--radius", type=int, help="the r value to set")
parser.add_argument("-n", "--nComponents", type=int, help="the number of components to use")


args = parser.parse_args()

#Generating model
if args.generateRawModel:
    path = args.model if args.model else "./models/rawModel.pkl"
    shared.generator.generateRawModel(path)

elif args.generatePCAModel:
    shared.generator.generatePCAModel(args.model)

elif args.generateReducedModel:
    if not args.nComponents:
        print("ERROR: You must specify n components to keep")
        shared.sys.exit(1)
    if not args.modelA:
        print("ERROR: You must specify the output model (modelA)")
        shared.sys.exit(1)
    shared.generator.generateReducedModel(args.modelA, args.nComponents)

#See & Modify a model
elif args.showModelSettings:
    shared.accessor.showModel(args.model)

elif args.setModelSettings:
    if not args.nComponents and not args.r:
        print("ERROR: You must either specify n components or a new radius")
        shared.sys.exit(1)
    shared.accessor.setModelSettings(args.model, args.nComponents, args.r)

#Benchmark a model
elif args.benchmark:
    shared.benchmark.benchmark(args.model)

elif args.benchmarkCompsNb:
    shared.benchmark.benchmarkCompsNb(args.model)

elif args.benchmarkRadius:
    shared.benchmark.benchmarkR(args.model)

#Compare two models
elif args.compareModels:
    if not args.modelA:
        print("Missing modelA argument")
        shared.sys.exit(1)
    if not args.modelB:
        print("Missing modelB argument")
    shared.benchmark.perfCompare(args.modelA, args.modelB)

#Helper
elif not len(shared.sys.argv) > 1 :
    parser.print_help()

shared.sys.exit()
