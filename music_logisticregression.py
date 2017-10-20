import pandas as pd
import numpy as np
import mb_logreg as lr

#Read training dataset for music genres. 
trainX = pd.read_csv('train.x.csv', sep=",")
trainY = pd.read_csv('train.y.csv', sep=",")

mean = []
standardDev = []
#Normalize data
def normalize(data):
        columns = data.ix[:, 1:].columns.values
        columns = columns[1:]

        for column in columns:
                mean.append(data[column].mean())
                standardDev.append(data[column].std())

 '''               for i, row in data.iterrows():

                        normVal = data.get_value(index=i, col=column)
                        normVal = (normVal - mean[len(mean)-1]) / standardDev[len(standardDev)-1]
                        data.set_value(index=i, col=column, value=normVal)
'''
        
normalize(trainX)

#Calculate number of labels/classes and order them.
labels = trainY.class_label.unique()
labels = sorted(labels)
m = len(labels) #number of class labels

index = np.arange(27) # array of numbers for the number of samples
#allWeights = pd.DataFrame(columns=columns, index = index, dtype = 'float')
allWeights = pd.DataFrame(data = np.zeros((27 , 13)), columns=labels)
print(allWeights)

#For each label change that label to 1 and all others to 0. Use these new values to create a hypothesis for that label.
for label in labels:
        print(label)

        tempY = trainY.copy()

        #For each row in tempY, change the label to either 0 or 1.
        for i, row in tempY.iterrows():
                binaryValue = 0

                if tempY.get_value(index=i, col='class_label') == label:
                        binaryValue = 1

                tempY.set_value(index=i, col='class_label', value=binaryValue)

        df = pd.DataFrame(tempY)
        df.to_csv('tempY.csv',sep=",",float_format='%10.8f',index=False)

        #With binary labels find best hypothesis.
        hypothesis = lr.findHypothesis(trainX.ix[:, 1:], tempY.ix[:, 1:])
        cost = lr.cost(trainX.ix[:, 1:], tempY.ix[:, 1:], hypothesis)
        print(cost)
        allWeights[label] = hypothesis[0].values
        
print(allWeights)
pred = lr.sigmoid(trainX.ix[:, 1:], allWeights)
print(pred)
df = pd.DataFrame(pred)
df.to_csv('predictionSubmit.csv',sep=",",float_format='%10.8f',index=False)


