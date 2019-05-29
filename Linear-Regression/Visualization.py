from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel, RidgeRegressionWithSGD, LassoWithSGD, RidgeRegressionModel, LassoModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
axLinear = fig.add_subplot(131, projection='3d')
axRidge = fig.add_subplot(132, projection='3d')
axLasso = fig.add_subplot(133, projection='3d')

def parsePoint(line):
    try:
        values = [float(x) for x in line.replace(',', ' ').split(' ')]
        return LabeledPoint(values[6], values[:6])
    except:
        pass

def parseVector(line):
    try:
        values = [float(x) for x in line.replace(',', ' ').split(' ')]
        return Vectors.dense(values[0], values[1], values[2], values[3], values[4], values[5])
    except:
        pass  

sc= SparkContext()
train_data = sc.textFile("train.csv")
test_data = sc.textFile("test.csv")

parsedTrainData = train_data.map(parsePoint).filter(lambda x: x is not None)
parsedTestData = test_data.map(parsePoint).filter(lambda x: x is not None)

mat = RowMatrix(train_data.map(parseVector).filter(lambda x: x is not None))
pc = mat.computePrincipalComponents(2)
projected = mat.multiply(pc)

x = [vector[0] for vector in projected.rows.collect()]
y = [vector[1] for vector in projected.rows.collect()]

LinearModel = LinearRegressionModel.load(sc, "Linear")
RidgeModel = RidgeRegressionModel.load(sc, "Ridge")
LassoModel = LassoModel.load(sc, "Lasso")

valuesAndPredsLinearTrain = parsedTrainData.map(lambda p: (p.label, LinearModel.predict(p.features)))
valuesAndPredsLinearTest = parsedTestData.map(lambda p: (p.label, LinearModel.predict(p.features)))

valuesAndPredsRidgeTrain = parsedTrainData.map(lambda p: (p.label, RidgeModel.predict(p.features)))
valuesAndPredsRidgeTest = parsedTestData.map(lambda p: (p.label, RidgeModel.predict(p.features)))

valuesAndPredsLassoTrain = parsedTrainData.map(lambda p: (p.label, LassoModel.predict(p.features)))
valuesAndPredsLassoTest = parsedTestData.map(lambda p: (p.label, LassoModel.predict(p.features)))

valueLinearTrain = [valueAndPred[0] for valueAndPred in valuesAndPredsLinearTrain.collect()]
predictLinearTrain = [valueAndPred[1] for valueAndPred in valuesAndPredsLinearTrain.collect()]

valueRidgeTrain = [valueAndPred[0] for valueAndPred in valuesAndPredsRidgeTrain.collect()]
predictRidgeTrain = [valueAndPred[1] for valueAndPred in valuesAndPredsRidgeTrain.collect()]

valueLassoTrain = [valueAndPred[0] for valueAndPred in valuesAndPredsLassoTrain.collect()]
predictLassoTrain = [valueAndPred[1] for valueAndPred in valuesAndPredsLassoTrain.collect()]

for marker, z in [('o', valueLinearTrain), ('^', predictLinearTrain)]:
    axLinear.scatter(x, y, z, marker=marker)

axLinear.set_xlabel('X Label')
axLinear.set_ylabel('Y Label')
axLinear.set_zlabel('Linear - House price of unit area')

for marker, z in [('o', valueRidgeTrain), ('^', predictLinearTrain)]:
    axRidge.scatter(x, y, z, marker=marker)

axRidge.set_xlabel('X Label')
axRidge.set_ylabel('Y Label')
axRidge.set_zlabel('Ridge - House price of unit area')

for marker, z in [('o', valueLassoTrain), ('^', predictLassoTrain)]:
    axLasso.scatter(x, y, z, marker=marker)

axLasso.set_xlabel('X Label')
axLasso.set_ylabel('Y Label')
axLasso.set_zlabel('Lasso - House price of unit area')

plt.show()


