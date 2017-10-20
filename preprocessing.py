import pandas as pd
import numpy as np

# constants
INPUT_FILE = 'train.y.csv'
OUTPUT_FILE = 'oneHotY.csv'


#Data to be modified
# Id,class_label
# 1,International
# 2,Vocal
# 3,Latin
# 4,Blues
# ...
trainY = pd.read_csv(INPUT_FILE, sep = ",")


#Change Y data which is 13000 x 2 dataframe into 13000 x 14 dataframe. Includes ID and transforming to one-hot encoding. 
#Each class will be a column and they are arranged alphabetically. 
labels = trainY.class_label.unique()
labels = sorted(labels)
#labels = np.insert(labels, 0, 'Id')

oneHotY = pd.DataFrame(data = np.zeros((13000 , 13)), columns=labels, dtype='int')

for label in labels:
    tempY = trainY.copy()

    #For each row in tempY, change the label to either 0 or 1.
    for i, row in tempY.iterrows():
        binaryValue = 0

        if tempY.get_value(index=i, col='class_label') == label:
            binaryValue = 1

        tempY.set_value(index=i, col='class_label', value=binaryValue)

    oneHotY[label] = tempY['class_label'].values

# Id,Blues,Country,Electronic,Folk,International,Jazz,Latin,New_Age,Pop_Rock,Rap,Reggae,RnB,Vocal
# 1,0,0,0,0,1,0,0,0,0,0,0,0,0
# 2,0,0,0,0,0,0,0,0,0,0,0,0,1
# 3,0,0,0,0,0,0,1,0,0,0,0,0,0
# ...
oneHotY.insert(loc = 0, column = 'Id', value = trainY['Id'])
oneHotY.to_csv(OUTPUT_FILE, sep=",", index=False)
