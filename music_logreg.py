#imports 
import pandas as pd
import numpy as np

# Hyperparameters
LEARNING_RATE = 0.1
REG = 0.001

# Constants

#Read in data
trainX = pd.read_csv('train.x.csv', sep=",")
oneHotY = pd.read_csv('oneHotY.csv', sep=",")
testX = pd.read_csv('test.x.csv', sep=",")

# Function definitions



# numExamples = trainX.shape[0]

#Normalize data.
mean = []
standardDev = []
def devNormalize(data):
    tempX = data.copy()
    # double check this?
    # columns = tempX.ix[:, 2:].columns.values
    columns = tempX.ix[:, 1:].columns.values
    columns = columns[1:]

    for column in columns:
        currentColumnMean = tempX[column].mean() 
        mean.append(currentColumnMean)
        standardDev.append(tempX[column].std())

        for i, row in tempX.iterrows():

            # maybe clearer? normVal = row.get_value(col=column) 
            tempX.get_value(index=i, col=column)

            normVal = (normVal - currentColumnMean) / standardDev[len(standardDev)-1]
            tempX.set_value(index=i, col=column, value=normVal)
    
    return tempX

normX = devNormalize(trainX)

def normalize(data):
    return (data.ix[:, 2:].values - mean) / standardDev


#Initialize weights
W = pd.DataFrame(data = 0.01 * np.random.randn(27,13))


#Gradient descent loop.
for i in range(500):

    #Compute class scores and normalize to get the probabilities for each class/genre.
    scores = np.dot(normX.ix[:, 1:], W)
    yHat = np.exp(scores) / np.sum(np.exp(scores), axis = 1, keepdims = True)


    #Compute loss with regularization
    correctYProb = np.sum(np.multiply(yHat, oneHotY.ix[:, 1:]), axis = 1)
    data_cost = np.mean(-np.log(correctYProb))
    reg_cost = 0.5 * REG * np.sum(np.sum(W * W))
    cost = data_cost + reg_cost
    if i % 10 == 0:
        print ("iteration %d: loss %f" % (i, cost))


    #Compute gradient
    dscores = (1/13000) * (yHat - oneHotY.ix[:, 1:])
    dW = np.dot(normX.ix[:, 1:].T, dscores)


    #Parameter update
    W += -LEARNING_RATE * dW

#MNormalize test data
normTestX = pd.DataFrame(normalize(testX))
normTestX.insert(loc = 0, column = 'Id', value = testX['Id'])
normTestX.insert(loc = 1, column = 'att0', value = testX['att0'])

#Make predictions with weights
prediction = np.dot(normTestX.ix[:, 1:], W)
prediction = np.exp(prediction) / np.sum(np.exp(prediction), axis = 1, keepdims = True)

predTest = pd.DataFrame(prediction)
predTest.insert(loc = 0, column = 'Id', value = testX['Id'])
predTest.columns = ['Id', 'Blues', 'Country', 'Electronic', 'Folk', 'International', 'Jazz', 
                    'Latin', 'New_Age', 'Pop_Rock', 'Rap', 'Reggae', 'RnB', 'Vocal']
predTest.to_csv('predictionSubmit.csv',sep=",",float_format='%.4f',index=False)