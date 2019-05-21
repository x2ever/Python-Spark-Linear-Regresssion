from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel, RidgeRegressionWithSGD, LassoWithSGD
import numpy as np

def parsePoint(line):
    try:
        values = [float(x) for x in line.replace(',', ' ').split(' ')]
        return LabeledPoint(values[6], values[:6])
    except:
        pass

sc= SparkContext()
train_data = sc.textFile("train.csv")
test_data = sc.textFile("test.csv")

parsedTrainData = train_data.map(parsePoint).filter(lambda x: x is not None)
parsedTestData = test_data.map(parsePoint).filter(lambda x: x is not None)

model_linear = LinearRegressionWithSGD.train(parsedTrainData, iterations=100, step=0.1)
model_ridge = RidgeRegressionWithSGD.train(parsedTrainData, iterations=100, step=0.1, regParam= 0.01)
model_lasso = LassoWithSGD.train(parsedTrainData, iterations=100, step=0.1, regParam= 0.01)

valuesAndPredsLinearTrain = parsedTrainData.map(lambda p: (p.label, model_linear.predict(p.features)))
valuesAndPredsLinearTest = parsedTestData.map(lambda p: (p.label, model_linear.predict(p.features)))

valuesAndPredsRidgeTrain = parsedTrainData.map(lambda p: (p.label, model_ridge.predict(p.features)))
valuesAndPredsRidgeTest = parsedTestData.map(lambda p: (p.label, model_ridge.predict(p.features)))

valuesAndPredsLassoTrain = parsedTrainData.map(lambda p: (p.label, model_lasso.predict(p.features)))
valuesAndPredsLassoTest = parsedTestData.map(lambda p: (p.label, model_lasso.predict(p.features)))

MSE_Linear = (valuesAndPredsLinearTrain.map(lambda vp: (vp[0] - vp[1])**2).reduce(lambda x, y: x + y) + valuesAndPredsLinearTest.map(lambda vp: (vp[0] - vp[1])**2).reduce(lambda x, y: x + y)) / \
    (valuesAndPredsLinearTrain.count() + valuesAndPredsLinearTest.count())

MSE_Ridge = (valuesAndPredsRidgeTrain.map(lambda vp: (vp[0] - vp[1])**2).reduce(lambda x, y: x + y) + valuesAndPredsRidgeTest.map(lambda vp: (vp[0] - vp[1])**2).reduce(lambda x, y: x + y)) / \
    (valuesAndPredsRidgeTrain.count() + valuesAndPredsRidgeTest.count())

MSE_Lasso = (valuesAndPredsLassoTrain.map(lambda vp: (vp[0] - vp[1])**2).reduce(lambda x, y: x + y) + valuesAndPredsLassoTest.map(lambda vp: (vp[0] - vp[1])**2).reduce(lambda x, y: x + y)) / \
    (valuesAndPredsLassoTrain.count() + valuesAndPredsLassoTest.count())

print("Root Mean Squared Error of 'Linear Model + Total Data Set': " + str(np.sqrt(MSE_Linear)))
print("Root Mean Squared Error of 'Ridge Model + Total Data Set': " + str(np.sqrt(MSE_Ridge)))
print("Root Mean Squared Error of 'Lasso Model + Total Data Set': " + str(np.sqrt(MSE_Lasso)))

model_linear.save(sc, "2014312051_linear")
model_ridge.save(sc, "2014312051_ridge")
model_lasso.save(sc, "2014312051_lasso")
