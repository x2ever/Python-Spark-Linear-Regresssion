from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel, RidgeRegressionWithSGD, LassoWithSGD
import numpy as np

def parsePoint(line):
    values = [float(x) for x in line.replace(',', ' ').split(' ')]
    return LabeledPoint(values[0], values[1:])

sc= SparkContext()
train_data = sc.textFile("train.csv")
test_data = sc.textFile("test.csv")

parsedTrainData = train_data.map(parsePoint).filter(lambda x: x is not None)
parsedTestData = test_data.map(parsePoint).filter(lambda x: x is not None)

model_lenear = LinearRegressionWithSGD.train(parsedTrainData, iterations=100, step=0.1)
model_ridge = RidgeRegressionWithSGD.train(parsedTrainData, iterations=100, step=0.1, regParam= 0.01)
model_lasso = LassoWithSGD.train(parsedTrainData, iterations=100, step=0.1, regParam= 0.01)

valuesAndPredsLinearTrain = parsedTrainData.map(lambda p: (p.label, model_lenear.predict(p.features)))
valuesAndPredsLinearTest = parsedTestData.map(lambda p: (p.label, model_lenear.predict(p.features)))

valuesAndPredsRidgeTrain = parsedTrainData.map(lambda p: (p.label, model_ridge.predict(p.features)))
valuesAndPredsRidgeTest = parsedTestData.map(lambda p: (p.label, model_ridge.predict(p.features)))

valuesAndPredsLassoTrain = parsedTrainData.map(lambda p: (p.label, model_lasso.predict(p.features)))
valuesAndPredsLassoTest = parsedTestData.map(lambda p: (p.label, model_lasso.predict(p.features)))

RMSE_Linear_Train = np.sqrt(valuesAndPredsLinearTrain.map(lambda vp: (vp[0] - vp[1])**2).reduce(lambda x, y: x + y) / valuesAndPredsLinearTrain.count())
RMSE_Linear_Test = np.sqrt(valuesAndPredsLinearTest.map(lambda vp: (vp[0] - vp[1])**2).reduce(lambda x, y: x + y) / valuesAndPredsLinearTest.count())

RMSE_Ridge_Train = np.sqrt(valuesAndPredsRidgeTrain.map(lambda vp: (vp[0] - vp[1])**2).reduce(lambda x, y: x + y) / valuesAndPredsRidgeTrain.count())
RMSE_Ridge_Test = np.sqrt(valuesAndPredsRidgeTest.map(lambda vp: (vp[0] - vp[1])**2).reduce(lambda x, y: x + y) / valuesAndPredsRidgeTest.count())

RMSE_Lasso_Train = np.sqrt(valuesAndPredsLassoTrain.map(lambda vp: (vp[0] - vp[1])**2).reduce(lambda x, y: x + y) / valuesAndPredsLassoTrain.count())
RMSE_Lasso_Test = np.sqrt(valuesAndPredsLassoTest.map(lambda vp: (vp[0] - vp[1])**2).reduce(lambda x, y: x + y) / valuesAndPredsLassoTest.count())

print("Root Mean Squared Error of 'Linear Model + Train Set': " + str(RMSE_Linear_Train))
print("Root Mean Squared Error of 'Linear Model + Test Set': " + str(RMSE_Linear_Test))

print("Root Mean Squared Error of 'Ridge Model + Train Set': " + str(RMSE_Ridge_Train))
print("Root Mean Squared Error of 'Ridge Model + Test Set': " + str(RMSE_Ridge_Test))

print("Root Mean Squared Error of 'Lasso Model + Train Set': " + str(RMSE_Lasso_Train))
print("Root Mean Squared Error of 'Lasso Model + Test Set': " + str(RMSE_Lasso_Test))

model_lenear.save(sc, "2014312051_linear")
model_ridge.save(sc, "2014312051_ridge")
model_lasso.save(sc, "2014312051_lasso")
