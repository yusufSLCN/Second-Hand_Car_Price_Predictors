import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


date_columns = ['dateCreated', 'lastSeen']
# A date looks like => '2016-04-07 03:16:57'
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
#raw = pd.read_csv('autos.csv', parse_dates=date_columns, date_parser=dateparse, encoding='cp1252')

#heads = list(raw.columns)


cleanData = raw.copy()
cleanData.drop(['nrOfPictures'], axis = 1, inplace = True)
cleanData.drop(['abtest'], axis = 1, inplace = True)
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
cleanData = cleanData[cleanData['price'] >= 250]
cleanData = cleanData[cleanData['price'] <= 100000]

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

        
cleanData.drop(['fuelType'], axis = 1, inplace = True)

valHeads = list(cleanData.columns)

trainHeads = list(cleanData.columns)
trainHeads.insert(0,'Bias')
#normalize data min-max normalization
featuresToNormalize = cleanData[['kilometer','yearOfRegistration', 'powerPS']]
#featuresToNormalize = (featuresToNormalize - featuresToNormalize.min()) / (featuresToNormalize.max() -featuresToNormalize.min())
featuresToNormalize = (featuresToNormalize - featuresToNormalize.mean()) / (featuresToNormalize.std())
cleanData[['kilometer','yearOfRegistration', 'powerPS']] = featuresToNormalize

cleanData = cleanData.to_numpy()

#------------------------------------------------------------------------------- validation - feature selection
dataSize = np.int(cleanData.shape[0])
np.random.seed(2)
idx = np.arange(dataSize)
np.random.shuffle( idx)
lamdha =  10
##feature selection
valD = cleanData[idx[:int(dataSize*15/100)]]
copyOfValD = np.copy(valD)
valPrice = price[idx[:int(dataSize*15/100)]]
numberOfSamples = np.int(valD.shape[0])
#
#ones = np.ones((numberOfSamples,1))
#
#tempFeat = ones
#optFeatNames = ['bias']
#
#optFeatIndex = []
#k = 20
#
#featIndex = list(range(int(valD.shape[1])))
#for feat in range(k):
#    fIndex = 0
#    minErr = float('inf')
#    print('Feature ' + str(feat))
#    for j in range(int(valD.shape[1]) - 1):
#        newFeat = valD[:,j].reshape(numberOfSamples, 1)
#        tempFeat = np.hstack((tempFeat,newFeat))
#        Brss = np.linalg.inv((tempFeat.T).dot(tempFeat) + lamdha*np.identity(tempFeat.shape[1])).dot(tempFeat.T).dot(valPrice);
#        #Brss = np.linalg.inv((tempFeat.T).dot(tempFeat)).dot(tempFeat.T).dot(trainPrice);
#        price_hat = tempFeat.dot(Brss);
#        
#        ## traninning error with mse
#        #trainMSE = 0;
#        #for i in range(numberOfSamples):
#        #    trainMSE = trainMSE + (price_hat[0,i] - price[0,i])^2 ;
#        #
#        #trainMSE = trainMSE/numberOfSamples;
#        #trainMSE = np.sqrt(trainMSE);
#        #print(trainMSE);
#
#        valErr = 0;
#        for i in range(numberOfSamples - 1):
#            valErr = valErr + np.absolute(price_hat[i,0] - valPrice[i,0])/valPrice[i,0];
#        
#        valErr = valErr / numberOfSamples * 100; 
#        
#        tempFeat = tempFeat[:,:-1]
#        if valErr < minErr:
#            minErr = valErr
#            fIndex = j
#            print(valErr);
#            
#    tempFeat = np.hstack((tempFeat,valD[:,fIndex].reshape(numberOfSamples, 1)))
#    valD = np.delete(valD,fIndex, 1)
#    
#    optFeatNames.append( valHeads[fIndex])
#    optFeatIndex.append(featIndex[fIndex])
#    del featIndex[fIndex]
#    del valHeads[fIndex]
#
#print('Validation Error')
#print(minErr)
#optFeatures = tempFeat
# ------------------------------------------------------------------------------------------------training 

trainD = cleanData[idx[int(dataSize*15/100):int(dataSize*85/100)]]
#trainD = trainD[:,np.asarray(optFeatIndex)] #uncomment for feature selecetion
trainD = np.hstack((np.ones((len(trainD),1)), trainD))

trainPrice = price[idx[int(dataSize*15/100):int(dataSize*85/100)]] 

#Linear Regression without regularization
#Brss = np.linalg.inv((trainD.T).dot(trainD)).dot(trainD.T).dot(trainPrice);

#Ridge
Brss = np.linalg.inv((trainD.T).dot(trainD) + lamdha*np.identity(trainD.shape[1])).dot(trainD.T).dot(trainPrice);
        
price_hat = trainD.dot(Brss);    
trainErr = 0;
for i in range(len(price_hat ) - 1):
    trainErr = trainErr + np.absolute(price_hat[i,0] - trainPrice[i,0])/trainPrice[i,0];

trainErr = trainErr / len(price_hat ) * 100; 
print('Training Error')
print(trainErr);

#--------------------------------------------------------------------------------------------------test

testD = cleanData[idx[int(dataSize*85/100):]]
#testD = testD[:,np.asarray(optFeatIndex)] #uncomment for feature selecetion
testD = np.hstack((np.ones((len(testD),1)), testD))

testPrice = price[idx[int(dataSize*85/100):]] 
        
testPriceHat = testD.dot(Brss);    
testErr = 0;
for i in range(len(testPriceHat ) - 1):
    testErr = testErr + np.absolute(testPriceHat[i,0] - testPrice[i,0])/testPrice[i,0];

testErr = testErr / len(testPriceHat ) * 100; 
print('Test Error')
print(testErr);

#with open('optFeatureInd.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#    pickle.dump(np.asarray(optFeatIndex), f)

## Getting back the objects:
#with open('optFeatureInd.pkl') as f:  # Python 3: open(..., 'rb')
#    obj0, obj1, obj2 = pickle.load(f)

