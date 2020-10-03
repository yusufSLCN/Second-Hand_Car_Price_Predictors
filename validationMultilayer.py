import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

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
cleanData.drop(['postalCode'], axis = 1, inplace = True
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

#model is removed 
#print('Rows without a model type', cleanData['model'].isna().sum())
#cleanData = cleanData.dropna(subset = ['model'])

print('Rows without a fuel type', cleanData['fuelType'].isna().sum())
cleanData = cleanData.dropna(subset = ['fuelType'])
cleanData = pd.concat([cleanData, pd.get_dummies(cleanData['fuelType'], prefix='fuelType')], axis=1)
cleanData.drop(['fuelType'], axis = 1, inplace = True)

#cleanData['fuelType'].value_counts().plot(kind='bar', title='Fuel type distribution')

#cleanData['brand'].value_counts().plot(kind='bar', figsize=(16, 8), title='Brand distribution of cars')

print('Rows without a notRepairedDamage', cleanData['notRepairedDamage'].isna().sum())
cleanData = cleanData.dropna(subset = ['notRepairedDamage'])
cleanData['isDamaged'] = np.where(cleanData['notRepairedDamage'] == 'ja', 1, 0)
cleanData.drop(['notRepairedDamage'], axis = 1, inplace = True)

#cleanData.plot(y='yearOfRegistration', kind='hist', bins=35, figsize=(10, 7), title='Cars and their registration years')
#raw.plot(y='price', kind='hist', figsize=(10, 7), bins=10, title='Km for cars...')
#pd.DataFrame.hist(raw,'price')

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

###select the correlated features
#corrFeat = featNames[ar[0,1:] - 1]
#cleanData = cleanData[:,ar[0,1:] - 1]
#-------------------------------------------- prepare training and test data
dataSize = np.int(cleanData.shape[0])
np.random.seed(2)
idx = np.arange(dataSize)
np.random.shuffle( idx)


featNames = np.asarray(featNames)
#get training, test and validation data
trainPrc = price[idx[:int(dataSize*75/100)]]
trainData = cleanData[idx[:int(dataSize*75/100)]]
testPrc = price[idx[int(dataSize*75/100):int(dataSize*90/100)]]
testData = cleanData[idx[int(dataSize*75/100):int(dataSize*90/100)]]
valPrc = price[idx[int(dataSize*90/100):]]
valData = cleanData[idx[int(dataSize*90/100):]]

#-------------------------------------------------------------------------------cross validation

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
epoch = 15
learnR = 2.7*10**-7
batchS = 120
#hidden layer sizes
N1 = 400
N2 = 200
N3 = 500

#K fold
K = 5

hiddenLay = 1
#validation
#valParts = np.vsplit(valData[:-(valLength% K)], K)

valData = np.append(valData, np.zeros([len(valData),1]) - 1, axis = 1)

#set the values that is going to be tried 
valValues = np.arange(100,500,100)
avrErrLog = []
for j in valValues:
    print('For number of hidden layer =' , j)
    avrValErr = 0
    N3 = j
    for valInd in range(K):
        #print('Fold K =', valInd)
        #prepare validation data for the next fold
        valLength = len(valData)
        valParts = np.arange(valLength)
        testIndex =  list(range(int(valLength/K) * valInd , int(valLength/K) * (valInd + 1)))
        valTestData = valData[testIndex]
        valTestPrc = valPrc[ testIndex]
        valTrainData = valData[ np.delete(valParts, testIndex )]
        valTrainPrc = valPrc[ np.delete(valParts, testIndex )]
        if hiddenLay == 1:
            wHidden1Val = np.random.normal(0, 0.1, (N1,len(valData[0])))
            wOutVal = np.random.normal(0, 0.1, (1, N1 + 1))
        elif hiddenLay == 2:
            wHidden1Val = np.random.normal(0, 0.1, (N1, len(valData[0])))
            wHidden2Val = np.random.normal(0, 0.1, (N2, N1 + 1))
            wOutVal = np.random.normal(0, 0.1, (1, N2 + 1))
        elif  hiddenLay == 3:
            wHidden1Val = np.random.normal(0, 0.1, (N1, len(valData[0])))
            wHidden2Val = np.random.normal(0, 0.1, (N2, N1 + 1))
            wHidden3Val = np.random.normal(0, 0.1, (N3, N2 + 1))
            wOutVal = np.random.normal(0, 0.1, (1, N3 + 1))

        for ep in range(epoch): 
            #print(ep)
    
            #--------------------------------------------------------------
            #shuffle the data each epoch for training
            shuffInd = np.arange(len(valTrainData))
            np.random.shuffle( shuffInd)
            valTrainPrc = valTrainPrc[shuffInd]
            valTrainData = valTrainData[shuffInd]
            
            for i in range(int(len(valTrainData)/batchS)):
                if hiddenLay == 1:
                    inp = valTrainData[i* batchS: i* batchS + batchS]
                    #first layer 
                    v1 = np.dot(wHidden1Val,inp.T)
                    h1 =  1/(1+np.exp(-v1))
                    h1 = np.append(h1, np.zeros([1,len(h1[0])]) - 1, axis = 0)
                    #output layer
                    vout = np.dot(wOutVal,h1)
                    trainOut = relu(vout)
                    
                elif hiddenLay == 2:
                    #first layer
                    inp = valTrainData[i* batchS: i* batchS + batchS]
                    v1 = np.dot(wHidden1Val, inp.T)
                    h1 =  1/(1+np.exp(-v1))#np.tanh(v1) 
                    h1 = np.append(h1, np.zeros([1,len(h1[0])]) - 1, axis = 0)
                    #second layer 
                    v2 = np.dot(wHidden2Val,h1)
                    h2 =  1/(1+np.exp(-v2))
                    h2 = np.append(h2, np.zeros([1,len(h2[0])]) - 1, axis = 0)
                    #output layer
                    vout = np.dot(wOutVal,h2)
                    trainOut = relu(vout)
                elif hiddenLay == 3:
                    #first layer
                    inp = valTrainData[i* batchS: i* batchS + batchS]
                    v1 = np.dot(wHidden1Val, inp.T)
                    h1 =  1/(1+np.exp(-v1))#np.tanh(v1) 
                    h1 = np.append(h1, np.zeros([1,len(h1[0])]) - 1, axis = 0)
                    #second layer 
                    v2 = np.dot(wHidden2Val,h1)
                    h2 =  1/(1+np.exp(-v2))
                    h2 = np.append(h2, np.zeros([1,len(h2[0])]) - 1, axis = 0)
                    #third layer
                    v3 = np.dot(wHidden3Val,h2)
                    h3 =  1/(1+np.exp(-v3))
                    h3 = np.append(h3, np.zeros([1,len(h3[0])]) - 1, axis = 0)
                    #output layer
                    vout = np.dot(wOutVal,h3)
                    trainOut = relu(vout)
                    
                #true label of the inputs
                dExp = valTrainPrc[i* batchS : i* batchS + batchS,:].T
                trainErr = np.sum((np.abs(dExp - trainOut)) / dExp) / len(dExp[0] )  * 100
                
                #calculate error of the batch
                err = dExp - trainOut

                #output layer back propagation   
                drelu = derOfRelu(vout) #(1,batch)
                deltaErr = err * drelu #(1,batch)
 
                if hiddenLay == 1:
                    dWout = learnR * np.dot(deltaErr, h1.T) #(1,batch)*(batch*N1 + 1)
                    #hidden layer back propogation 
                    hidden1Err = np.dot(deltaErr.T, wOutVal[:,0:-1]) #(batch,1) * (1,N1)
                    dsig = ( 1 - h1[0:-1]) * h1[0:-1]# 1 - h1[0:-1] * h1[0:-1] #(N1, batch)
                    deltaHidden1Err = hidden1Err.T * dsig #(N1 , batch)
                    dW1hidden = learnR * np.dot(deltaHidden1Err, inp) #(N1, batch) * (batch, inputLength)
                    
                    wHidden1Val += dW1hidden
                    wOutVal += dWout
                    
                elif hiddenLay == 2:
                    dWout = learnR * np.dot(deltaErr, h2.T) 
                    #second hidden layer back propogation 
                    hidden2Err = np.dot(deltaErr.T, wOutVal[:,0:-1]) 
                    dsig = ( 1 - h2[0:-1]) * h2[0:-1]
                    deltaHidden2Err = hidden2Err.T * dsig 
                    dW2hidden = learnR * np.dot(deltaHidden2Err, h1.T) 
                    #hidden layer back propagation   
                    hidden1Err = np.dot(deltaHidden2Err.T, wHidden2Val[:,0:-1])
                    dsig1 = ( 1 - h1[0:-1]) * h1[0:-1]
                    deltaHidden1Err = hidden1Err.T * dsig1
                    dW1hidden = learnR * np.dot(deltaHidden1Err, inp)                
                    #update weights
                    wHidden1Val += dW1hidden
                    wHidden2Val += dW2hidden
                    wOutVal += dWout
                    
                elif hiddenLay == 3:
                    dWout = learnR * np.dot(deltaErr, h3.T) 
                    #third hidden layer back propogation 
                    hidden3Err = np.dot(deltaErr.T, wOutVal[:,0:-1]) 
                    dsig = ( 1 - h3[0:-1]) * h3[0:-1]
                    deltaHidden3Err = hidden3Err.T * dsig 
                    dW3hidden = learnR * np.dot(deltaHidden3Err, h2.T) 
                    #second hidden layer back propagation   
                    hidden2Err = np.dot(deltaHidden3Err.T, wHidden3Val[:,0:-1])
                    dsig1 = ( 1 - h2[0:-1]) * h2[0:-1]
                    deltaHidden2Err = hidden2Err.T * dsig1
                    dW2hidden = learnR * np.dot(deltaHidden2Err, h1.T)  
                    #hidden layer back propagation   
                    hidden1Err = np.dot(deltaHidden2Err.T, wHidden2Val[:,0:-1])
                    dsig2 = ( 1 - h1[0:-1]) * h1[0:-1]
                    deltaHidden1Err = hidden1Err.T * dsig2
                    dW1hidden = learnR * np.dot(deltaHidden1Err, inp) #(N1, batch) * (batch, inputLength)    
                    
                    #update weights
                    wHidden1Val += dW1hidden
                    wHidden2Val += dW2hidden
                    wHidden3Val += dW3hidden
                    wOutVal += dWout
                
        #calculate validation error after each validation set
        valInp = valTestData
        if hiddenLay == 1:
            #fist layer
            valV1 = np.dot(wHidden1Val, valInp.T)
            valH1 = 1/(1+np.exp(-valV1))#np.tanh(valV1) 
            valH1 = np.append(valH1, np.zeros([1,len(valH1[0])]) - 1, axis = 0)
            #output layer
            valOut = np.dot(wOutVal,valH1)
            valOut = relu(valOut).T
            
        elif hiddenLay  == 2:
            #fist layer
            valV1 = np.dot(wHidden1Val, valInp.T)
            valH1 = 1/(1+np.exp(-valV1))#np.tanh(valV1) 
            valH1 = np.append(valH1, np.zeros([1,len(valH1[0])]) - 1, axis = 0)
            #second layer
            valV2 = np.dot(wHidden2Val, valH1)
            valH2 = 1/(1+np.exp(-valV2))
            valH2 = np.append(valH2, np.zeros([1,len(valH2[0])]) - 1, axis = 0)
            #output layer
            valOut = np.dot(wOutVal,valH2)
            valOut = relu(valOut).T
            
        elif hiddenLay  == 3:
            #fist layer
            valV1 = np.dot(wHidden1Val, valInp.T)
            valH1 = 1/(1+np.exp(-valV1))#np.tanh(valV1) 
            valH1 = np.append(valH1, np.zeros([1,len(valH1[0])]) - 1, axis = 0)
            #second layer
            valV2 = np.dot(wHidden2Val, valH1)
            valH2 = 1/(1+np.exp(-valV2))
            valH2 = np.append(valH2, np.zeros([1,len(valH2[0])]) - 1, axis = 0)
            #third layer 
            valV3 = np.dot(wHidden3Val, valH2)
            valH3 = 1/(1+np.exp(-valV3))
            valH3 = np.append(valH3, np.zeros([1,len(valH3[0])]) - 1, axis = 0)
            #output layer
            valOut = np.dot(wOutVal,valH3)
            valOut = relu(valOut).T
        
        bacthValErr = np.sum((np.abs(valOut - valTestPrc)) / valTestPrc) / len(valTestPrc )  * 100
        avrValErr = avrValErr + bacthValErr
    avr = avrValErr / K
    
    print('Average val error',avr)
    
    avrErrLog.append(avr)
plt.plot(valValues[:len(avrErrLog)], avrErrLog)
plt.title('Percentage error by the first hidden layer size')
plt.ylabel('%')
plt.xlabel('hidden layer size')