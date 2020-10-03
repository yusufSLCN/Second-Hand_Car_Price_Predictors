import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

plt.close("all")
#date_columns = ['dateCreated', 'lastSeen']
#dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
#raw = pd.read_csv('autos.csv', parse_dates=date_columns, date_parser=dateparse, encoding='cp1252')

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
print('Rows without a vehicle type', cleanData['vehicleType'].isna().sum())
print('Total number of rows', len(cleanData))
cleanData = cleanData.dropna(subset = ['vehicleType'])
print('Rows after droping cars without vehicle type', len(cleanData))

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

cleanData.drop(['brand'], axis = 1, inplace = True)

cleanData = pd.concat([cleanData, pd.get_dummies(cleanData['vehicleType'], prefix='vehicleType')], axis=1)
cleanData.drop(['vehicleType'], axis = 1, inplace = True)

cleanData = cleanData.dropna(subset = ['model'])
cleanData = pd.concat([cleanData, pd.get_dummies(cleanData['model'], prefix='model')], axis=1)
cleanData.drop(['model'], axis = 1, inplace = True)


#store price in another array
price = cleanData['price'].to_numpy()[:, np.newaxis]
cleanData.drop(['price'], axis = 1, inplace = True)


featNames = list(cleanData.columns)
#normalize data min-max normalization
featuresToNormalize = cleanData[['kilometer','yearOfRegistration', 'powerPS']]
#featuresToNormalize = (featuresToNormalize - featuresToNormalize.min()) / (featuresToNormalize.max() -featuresToNormalize.min())
featuresToNormalize = (featuresToNormalize - featuresToNormalize.mean()) / (featuresToNormalize.std())
#featuresToNormalize = (featuresToNormalize - featuresToNormalize.min()) / (featuresToNormalize.max() -featuresToNormalize.min()) - 0.5
cleanData[['kilometer','yearOfRegistration', 'powerPS']] = featuresToNormalize

cleanData = cleanData.to_numpy()
#PCA
#Number_of_dimensions = 120
## calculate covariance matrix of centered matrix
#CovMatrice = np.cov(cleanData.T)
## eigendecomposition of covariance matrix
#values, vectors = np.linalg.eig(CovMatrice)
## highest eigenvalue vectors
#new_order = (-values).argsort()[:Number_of_dimensions]
#new_vectors = vectors[new_order]
#cleanData = np.dot(cleanData,new_vectors.T)
#featNames = np.asarray(featNames)
##select the correlated features- not good results
# Find correleation coeff and sort

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

#----------------------------------------------------------------------training

def relu(x):
    y = np.copy(x)
    y[ y < 0 ] = 0
    return y

def derOfRelu(x):
    y = np.copy(x)
    y[ y < 0 ] = 0
    y[ y >= 0 ] = 1
    return y
#initilize the parameters
epoch = 30
learnR = 2.7*10**-7
batchS = 120
N = 400

#add -1 column to the last
trainData = np.append(trainData, np.zeros([len(trainData),1]) - 1, axis = 1)
#embed bias into weights by trainData + 1
wHidden = np.random.normal(0, 0.1, (N, len(trainData[0] + 1) ))
wOut = np.random.normal(0, 0.1, (1, N + 1))
valErrLog = []
trainErrLog = []
t = time.time()
#training 
for ep in range(epoch): 
    print('epoch: ' + str(ep))
    #test the network before each epoch
    valInp = valData
    valInp = np.append(valInp, np.zeros([len(valInp),1]) - 1, axis = 1)
    valV1 = np.dot(wHidden, valInp.T)
    valH1 = 1/(1+np.exp(-valV1))
    valH1 = np.append(valH1, np.zeros([1,len(valH1[0])]) - 1, axis = 0)
    #second layer
    valV2 = np.dot(wOut,valH1)
    valOut = relu(valV2).T
    
    #Percent err of test output
    valErr = np.sum((np.abs(valPrc - valOut)) / valPrc) / len(valPrc )  * 100
    valErrLog.append(valErr)

    trainErr = 0
    #--------------------------------------------------------------
    #shuffle the data each epoch for training
    shuffInd = np.arange(len(trainData))
    np.random.shuffle( shuffInd)
    trainPrc = trainPrc[shuffInd]
    trainData = trainData[shuffInd]
    for i in range(int(len(trainData)/batchS)):
        #first layer
        inp = trainData[i* batchS: i* batchS + batchS]
        v1 = np.dot(wHidden, inp.T)
        h1 =  1/(1+np.exp(-v1))
        h1 = np.append(h1, np.zeros([1,len(h1[0])]) - 1, axis = 0)
        #output layer
        v2 = np.dot(wOut,h1)
        trainOut = relu(v2)
        
        #true label of the inputs
        dExp = trainPrc[i* batchS : i* batchS + batchS,:].T
        
        trainErr += np.sum((np.abs(dExp - trainOut)) / dExp) / len(dExp[0] )  * 100
        
        #calculate error of the batch
        err = dExp - trainOut
        
        #output layer back propagation   
        drelu = derOfRelu(v2)
        deltaErr = err * drelu 
        dWout = learnR * np.dot(deltaErr, h1.T) 
        #hidden layer back propagation   
        hiddenErr = np.dot(deltaErr.T, wOut[:,0:-1]) 
        dsig = ( 1 - h1[0:-1]) * h1[0:-1]
        deltaHiddenErr = hiddenErr.T * dsig 
        dWhidden = learnR * np.dot(deltaHiddenErr, inp)
        #update weights
        wHidden += dWhidden
        wOut += dWout
    print('Train Error')
    print(trainErr/ int(len(trainData)/batchS));
    trainErrLog.append(trainErr/ int(len(trainData)/batchS))
    
elapsed = time.time() - t   
print('Training lasts ', elapsed, 's')  
print('Training err ', trainErrLog[-1])
plt.figure(3)  
plt.plot(range(1,epoch +1 ), trainErrLog)
plt.ticklabel_format(axis='x',style='sci',scilimits=(0,3))
plt.title('Training error through epochs')
plt.ylabel('%')
plt.xlabel('Epochs')

plt.figure(4)  
plt.plot(range(1,epoch +1 ), valErrLog)
plt.ticklabel_format(axis='x',style='sci',scilimits=(0,3))
plt.title('Validation error through epochs')
plt.ylabel('%')
plt.xlabel('Epochs')
print('Validation Error')
print(valErrLog[-1]);
#---------------------------------------------------------------------test
testInp = testData
testInp = np.append(testInp, np.zeros([len(testInp),1]) - 1, axis = 1)
testV1 = np.dot(wHidden, testInp.T)
testH1 = 1/(1+np.exp(-testV1))#np.tanh(testV1) 
testH1 = np.append(testH1, np.zeros([1,len(testH1[0])]) - 1, axis = 0)
#second layer
testV2 = np.dot(wOut,testH1)
testOut = relu(testV2).T

#Percentage error of test output
testErr = np.sum((np.abs(testPrc - testOut)) / testPrc) / len(testPrc )  * 100
print('Test Error')
print(testErr);

