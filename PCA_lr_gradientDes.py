import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

plt.close("all")
date_columns = ['dateCreated', 'lastSeen']
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
raw = pd.read_csv('autos.csv', parse_dates=date_columns, date_parser=dateparse, encoding='cp1252')

cleanData = raw.copy()
cleanData.drop(['nrOfPictures'], axis = 1, inplace = True)
cleanData.drop(['abtest'], axis = 1, inplace = True)
cleanData = cleanData[cleanData.seller != 'gewerblich']
cleanData.drop(['seller'], axis = 1, inplace = True)

cleanData.drop(['offerType'], axis = 1, inplace = True)
cleanData.drop(['dateCrawled'], axis = 1, inplace = True)
cleanData.drop(['dateCreated'], axis = 1, inplace = True)
cleanData.drop(['lastSeen'], axis = 1, inplace = True)
cleanData.drop(['postalCode'], axis = 1, inplace = True)
cleanData.drop(['monthOfRegistration'], axis = 1, inplace = True)
# todo try with these features
cleanData.drop(['name'], axis = 1, inplace = True)


#discarding meaningless data
cleanData = cleanData[cleanData['powerPS'] >= 60]
cleanData = cleanData[cleanData['powerPS'] <= 1000]
cleanData = cleanData[cleanData['yearOfRegistration'] > 1910]
cleanData = cleanData[cleanData['yearOfRegistration'] < 2019]
minP, maxP = 250, 100000
cleanData = cleanData[cleanData['price'] >= minP]
cleanData = cleanData[cleanData['price'] <= maxP]

#removing data points with null feature
print('Total number of rows', len(cleanData))
print('Rows without a vehicle type', cleanData['vehicleType'].isna().sum())
cleanData = cleanData.dropna(subset = ['vehicleType'])

print('Rows without a gearbox type', cleanData['gearbox'].isna().sum())
cleanData = cleanData.dropna(subset = ['gearbox'])
cleanData['gearbox'] = np.where(cleanData['gearbox'] == 'manuell', 1, 0)

print('Rows without a fuel type', cleanData['fuelType'].isna().sum())
cleanData = cleanData.dropna(subset = ['fuelType'])
cleanData = pd.concat([cleanData, pd.get_dummies(cleanData['fuelType'], prefix='fuelType')], axis=1)
cleanData.drop(['fuelType'], axis = 1, inplace = True)


print('Rows without a notRepairedDamage', cleanData['notRepairedDamage'].isna().sum())
cleanData = cleanData.dropna(subset = ['notRepairedDamage'])
cleanData['isDamaged'] = np.where(cleanData['notRepairedDamage'] == 'ja', 1, 0)
cleanData.drop(['notRepairedDamage'], axis = 1, inplace = True)



#one hot encoding 
cleanData = pd.concat([cleanData, pd.get_dummies(cleanData['brand'], prefix='brand')], axis=1)
#cleanData = cleanData[cleanData['brand'] == 'ford']
print('Rows without a brand', cleanData['brand'].isna().sum())
cleanData.drop(['brand'], axis = 1, inplace = True)


print('Rows without a vehicleType', cleanData['vehicleType'].isna().sum())
cleanData = pd.concat([cleanData, pd.get_dummies(cleanData['vehicleType'], prefix='vehicleType')], axis=1)
cleanData.drop(['vehicleType'], axis = 1, inplace = True)


cleanData = pd.concat([cleanData, pd.get_dummies(cleanData['model'], prefix='model')], axis=1)
cleanData.drop(['model'], axis = 1, inplace = True)



#store price in another array
price = cleanData['price'].to_numpy()[:, np.newaxis]


#featureNames
featNames = list(cleanData.columns)

featuresToNormalize = cleanData[['kilometer','yearOfRegistration', 'powerPS']]
#normalize data min-max normalization
#featuresToNormalize = (featuresToNormalize - featuresToNormalize.min()) / (featuresToNormalize.max() -featuresToNormalize.min())
#standatize
featuresToNormalize = (featuresToNormalize - featuresToNormalize.mean()) / (featuresToNormalize.std())
#featuresToNormalize = (featuresToNormalize - featuresToNormalize.min()) / (featuresToNormalize.max() -featuresToNormalize.min()) - 0.5
cleanData[['kilometer','yearOfRegistration', 'powerPS']] = featuresToNormalize


cleanData = cleanData.to_numpy()




print('Remaining data points after cleaning data:', cleanData.shape[0])
print('Number of features:', cleanData.shape[1] - 1)
featNames = np.asarray(featNames)
#--------------------------------------------------------------select the features 
## Find correleation coeff and sort
corrCoef = np.corrcoef(cleanData.T)

#find features that are correlated more than 0.1
ar = np.asarray(np.where(np.abs(corrCoef[0]) > 0.1))
corrPrcCoef = np.abs(corrCoef[0, ar])
corrFeat = featNames[ar]
#
plt.figure(3)
#shorten the feature names
plt.bar(range(len(corrPrcCoef[0])),corrPrcCoef[0])
for j in range(len(corrFeat[0])):
    if len(corrFeat[0,j]) > 10:
        corrFeat[0,j] = corrFeat[0,j][6:]
       
        
plt.xticks(range(len(corrPrcCoef[0])), corrFeat[0],rotation=90)
plt.xlabel('Highly correlated features')
plt.ylabel('Correlation coefficient')
plt.title('Features that are highy correlated with price')
plt.tight_layout()

##select the highly correlated features 
cleanData = cleanData[:,ar[0]]
#remove prices
cleanData = cleanData[:,1:]


#PCA
Number_of_dimensions = 20
# calculate covariance matrix of centered matrix
CovMatrice = np.cov(cleanData.T)
# eigendecomposition of covariance matrix
values, vectors = np.linalg.eig(CovMatrice)
# highest eigenvalue vectors
new_order = (-values).argsort()[:Number_of_dimensions]
new_vectors = vectors[new_order]
cleanData = np.dot(cleanData,new_vectors.T)

#-------------------------------------------- prepare training and test data
dataSize = np.int(cleanData.shape[0])
np.random.seed(2)
idx = np.arange(dataSize)
np.random.shuffle( idx)

#get training, test and validation data
trainPrc = price[idx[:int(dataSize*75/100)]]
trainData = cleanData[idx[:int(dataSize*75/100)]]
testPrc = price[idx[int(dataSize*75/100):int(dataSize*90/100)]]
testData = cleanData[idx[int(dataSize*75/100):int(dataSize*90/100)]]
valPrc = price[idx[int(dataSize*90/100):]]
valData = cleanData[idx[int(dataSize*90/100):]]

testData = np.hstack((np.ones((len(testData),1)), testData))
trainData = np.hstack((np.ones((len(trainData),1)), trainData))
valData = np.hstack((np.ones((len(valData),1)), valData))
#---------------------------------------------------------- validation
##K fold
#K = 5
#validation
valLRate = 2.7*10**-7
valValues = np.arange(10,70, 10)
avrErrLog = []
#PCA for validation data
Number_of_dimensions = 20
# calculate covariance matrix of centered matrix
CovMatrice = np.cov(valData.T)
# eigendecomposition of covariance matrix
values, vectors = np.linalg.eig(CovMatrice)
# highest eigenvalue vectors
new_order = (-values).argsort()[:Number_of_dimensions]
new_vectors = vectors[new_order]
valDataPCA = np.dot(valData,new_vectors.T)
    
epoch = 30
for j in valValues:
    print('For N =' , j)
    avrValErr = 0
    epoch = j
    for valInd in range(K):
        #print('Fold K =', valInd)
        valLength = len(valDataPCA)
        valParts = np.arange(valLength)
        testIndex =  list(range(int(valLength/K) * valInd , int(valLength/K) * (valInd + 1)))
        valTestData = valDataPCA[testIndex]
        valTestPrc = valPrc[ testIndex]
        valTrainData = valDataPCA[ np.delete(valParts, testIndex )]
        valTrainPrc = valPrc[ np.delete(valParts, testIndex )]
        
        valWeights = np.zeros([ len(valTrainData[0]), 1])
        for ep in range(epoch): 
            #print(ep)
            #shuffle the data each epoch for training
            shuffInd = np.arange(len(valTrainData))
            np.random.shuffle( shuffInd)
            valTrainPrc = valTrainPrc[shuffInd]
            valTrainData = valTrainData[shuffInd]
            #epochErr = 0
            for i in range(len(valTrainData)):
                valPred = np.dot(valTrainData[i],valWeights)
                err = valTrainPrc[i] - valPred
                dWeights =  err * valTrainData[i]#np.sum( err * trainData, axis  = 0)[np.newaxis].T
                valWeights += valLRate * dWeights[np.newaxis].T
                #epochErr += np.abs(err)/ valTrainPrc[i] * 100
                
        #calculate validation error after each validation set
        valTestPrcPred = np.dot(valTestData,valWeights)
        testErr = np.abs(valTestPrc - valTestPrcPred) /valTestPrc 
        avrFoldErr = np.sum(testErr, axis = 0) / len(valTestData) * 100
        print('Val Test error: ', avrFoldErr)
        avrValErr += avrFoldErr

        
    avr = avrValErr / K
    
    print('Average val error',avr)
    
    avrErrLog.append(avr)
    
plt.figure(2)
plt.plot(valValues[:len(avrErrLog)], avrErrLog)
plt.ticklabel_format(axis='x',style='sci',scilimits=(0,3))
plt.title('Validation percentage error by trained epochs')
plt.ylabel('%')
plt.xlabel('Epochs')

#----------------------------------------------- train model
t = time.time()
epochL = 3
lRate = 6*10**-7
trainErrLog = []
weights = np.zeros([ len(trainData[0]), 1])
for ep in range(epochL):
    avrErr = 0
    shuffInd = np.arange(len(trainData))
    np.random.shuffle( shuffInd)
    trainPrc = trainPrc[shuffInd]
    trainData = trainData[shuffInd]
    
    for i in range(len(trainData)):
        trainPred = np.dot(trainData[i],weights)
        err = trainPrc[i] - trainPred
        dWeights =  err * trainData[i]
        weights += lRate * dWeights[np.newaxis].T
        avrErr += np.abs(err)/ trainPrc[i] * 100
    
    print('Epoch', ep , 'error: ', np.sum(avrErr )/ len(trainData))
    trainErrLog.append(np.sum(avrErr )/ len(trainData))
elapsed = time.time() - t
print(elapsed, 's passed')
plt.figure(4)  
plt.plot(range(1,epochL +1 ), trainErrLog)
plt.ticklabel_format(axis='x',style='sci',scilimits=(0,3))
plt.title('Training percentage error through epochs')
plt.ylabel('%')
plt.xlabel('Epochs')

#---------------------------------------------------validate model
valPrcPred = np.dot(valData,weights)
valErr = np.abs(valPrc - valPrcPred) /valPrc 
avrValErr = np.sum(valErr, axis = 0) / len(valData) * 100
print('Validation error: ', avrValErr)

#---------------------------------------------------test model
testPrcPred = np.dot(testData,weights)
testErr = np.abs(testPrc - testPrcPred) /testPrc 
avrTestErr = np.sum(testErr, axis = 0) / len(testData) * 100
print('Test error: ', avrTestErr)