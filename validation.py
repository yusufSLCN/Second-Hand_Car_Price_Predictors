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
featNames = np.asarray(featNames)
###select the correlated features
#corrFeat = featNames[ar[0,1:] - 1]
#cleanData = cleanData[:,ar[0,1:] - 1]
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

#-------------------------------------------------------------------------------validation

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


#K fold
K = 5

#validation
#valParts = np.vsplit(valData[:-(valLength% K)], K)

valData = np.append(valData, np.zeros([len(valData),1]) - 1, axis = 1)
valValues = np.arange(20,200, 20)
avrErrLog = []

for j in valValues:
    print('For N =' , j)
    avrValErr = 0
    Number_of_dimensions = j
    N =j
    for valInd in range(K):
        
        #print('Fold K =', valInd)
        
        valLength = len(valData)
        valParts = np.arange(valLength)
        testIndex =  list(range(int(valLength/K) * valInd , int(valLength/K) * (valInd + 1)))
        valTestData = valData[testIndex]
        valTestPrc = valPrc[ testIndex]
        valTrainData = valData[ np.delete(valParts, testIndex )]
        valTrainPrc = valPrc[ np.delete(valParts, testIndex )]
        
        

    
        
        wHiddenVal = np.random.normal(0, 0.1, (N, len(valData[0])))
        wOutVal = np.random.normal(0, 0.1, (1, N+ 1))
        for ep in range(epoch): 
            #print(ep)
    
            #--------------------------------------------------------------
            #shuffle the data each epoch for training
            shuffInd = np.arange(len(valTrainData))
            np.random.shuffle( shuffInd)
            valTrainPrc = valTrainPrc[shuffInd]
            valTrainData = valTrainData[shuffInd]
            
            for i in range(int(len(valTrainData)/batchS)):
                #first layer
                inp = valTrainData[i* batchS: i* batchS + batchS]
                v1 = np.dot(wHiddenVal, inp.T)
                h1 =  1/(1+np.exp(-v1))#np.tanh(v1) 
                h1 = np.append(h1, np.zeros([1,len(h1[0])]) - 1, axis = 0)
                #output layer
                v2 = np.dot(wOutVal,h1)
                trainOut = relu(v2)
                
                #true label of the inputs
                dExp = valTrainPrc[i* batchS : i* batchS + batchS,:].T
                
                trainErr = np.sum((np.abs(dExp - trainOut)) / dExp) / len(dExp[0] )  * 100
        #        print('Train Error: '+ str(trainErr))
                #calculate error of the batch
                err = dExp - trainOut

                #output layer back propagation   
                drelu = derOfRelu(v2) #(1,batch)
                deltaErr = err * drelu #(1,batch)
                dWout = learnR * np.dot(deltaErr, h1.T) #(1,batch)*(batch*N1 + 1)
                #hidden layer back propagation   
                hiddenErr = np.dot(deltaErr.T, wOutVal[:,0:-1]) #(batch,1) * (1,N1)
                dsig = ( 1 - h1[0:-1]) * h1[0:-1]# 1 - h1[0:-1] * h1[0:-1] #(N1, batch)
                deltaHiddenErr = hiddenErr.T * dsig #(N1 , batch)
                dWhidden = learnR * np.dot(deltaHiddenErr, inp) #(N1, batch) * (batch, inputLength)
                #update weights
                wHiddenVal += dWhidden
                wOutVal += dWout
                
        #calculate validation error after each validation set
        valInp = valTestData
#        valInp = np.append(valInp, np.zeros([len(valInp),1]) - 1, axis = 1)
        valV1 = np.dot(wHiddenVal, valInp.T)
        valH1 = 1/(1+np.exp(-valV1))#np.tanh(valV1) 
        valH1 = np.append(valH1, np.zeros([1,len(valH1[0])]) - 1, axis = 0)
        #second layer
        valV2 = np.dot(wOutVal,valH1)
        valOut = relu(valV2).T
        
        bacthValErr = np.sum((np.abs(valOut - valTestPrc)) / valTestPrc) / len(valTestPrc )  * 100
        avrValErr = avrValErr + bacthValErr
    avr = avrValErr / K
    
    print('Average val error',avr)
    
    avrErrLog.append(avr)
plt.plot(valValues[:len(avrErrLog)], avrErrLog)
plt.title('Percentage Error by the bacth size')
plt.ylabel('%')
plt.xlabel('bacth size')