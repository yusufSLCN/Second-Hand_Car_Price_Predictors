# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 00:34:01 2019

@author: Burak Can
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

plt.close("all")
date_columns = ['dateCreated', 'lastSeen']
# A date looks like => '2016-04-07 03:16:57'
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
raw = pd.read_csv('autos.csv', parse_dates=date_columns, date_parser=dateparse, encoding='cp1252')

heads = list(raw.columns)


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

cleanData = cleanData[:,1:]


#%%

Number_of_dimensions = 15
# calculate the mean of each column
featureMeans = np.mean(cleanData.T, axis=1)
# center columns by subtracting column means
Centered_data = cleanData #- featureMeans
# calculate covariance matrix of centered matrix
CovMatrice = np.cov(Centered_data.T)
# eigendecomposition of covariance     
values, vectors = np.linalg.eig(CovMatrice)
# highest eigenvalue vectors
new_order = (-values).argsort()[:Number_of_dimensions]
new_vectors = vectors[new_order]

pca_outcome = np.dot(Centered_data,new_vectors.T)

#%%
dataSize = np.int(pca_outcome.shape[0])
np.random.seed(2)
idx = np.arange(dataSize)
np.random.shuffle( idx)

#get training, test and validation data
trainPrc = price[idx[:int(dataSize*75/100)]]
trainData = pca_outcome[idx[:int(dataSize*75/100)]]
testPrc = price[idx[int(dataSize*75/100):int(dataSize*90/100)]]
testData = pca_outcome[idx[int(dataSize*75/100):int(dataSize*90/100)]]
valPrc = price[idx[int(dataSize*90/100):]]
valData = pca_outcome[idx[int(dataSize*90/100):]]

testData = np.hstack((np.ones((len(testData),1)), testData))
trainData = np.hstack((np.ones((len(trainData),1)), trainData))
valData = np.hstack((np.ones((len(valData),1)), valData))



#%%



class RandomTree():
    def __init__(self, x, y, featureNumber, feature_indices, data_indices, max_depth ,min_node_sample):
        
        self.x = x
        self.y = y 
        self.featureNumber = featureNumber
        self.data_indices = data_indices
        self.feature_indices = feature_indices
        self.max_depth = max_depth 
        self.min_node_sample = min_node_sample;
        self.length = self.data_indices.shape[0]        
        self.avg_price = np.mean(y[data_indices]) ###################
        if np.isnan(self.avg_price):
            print("nan  avg_price")
            self.avg_price=0
        if  self.length == 1:
            print("one")
        self.weighted_sd = float('inf')
        for j in self.feature_indices: 
            temp_x = self.x[self.data_indices, j ]
            temp_y = self.y[self.data_indices]                    
            sorted_index = np.argsort(temp_x,axis=0) ###
            sorted_x, sorted_y = temp_x[sorted_index], temp_y[sorted_index] 
            rightCount,rightSum,rightSumSquare = self.length, sorted_y.sum(),(sorted_y**2).sum()
            leftCount,leftSum,leftSumSquare = 0,0.,0.
                            
            for i in range(0,self.length-self.min_node_sample-1):    
                leftCount += 1; 
                rightCount -= 1
                leftSum += sorted_y[i] 
                rightSum -= sorted_y[i] 
                leftSumSquare += sorted_y[i]**2 
                rightSumSquare -= sorted_y[i]**2                 
                if i<self.min_node_sample or sorted_x[i]==sorted_x[i+1]:
                    continue
#                if leftCount == 0 or rightCount==0 :     
#                    current_weighted_sd = float('inf')
#                else:     
                    
                left_Sd = ((leftSumSquare/leftCount) - (leftSum/leftCount)**2)**0.5 
                right_Sd = ((rightSumSquare/rightCount) - (rightSum/rightCount)**2)**0.5                
                current_weighted_sd = left_Sd*leftCount + right_Sd*rightCount
                    
                if current_weighted_sd<self.weighted_sd: 
                    self.selected_feature,self.weighted_sd,self.decision_value = j,current_weighted_sd,sorted_x[i] #change this      
        if self.weighted_sd== float('inf') or self.max_depth <= 0:
            return
        temp_right_fi = np.arange(self.x.shape[1])
        np.random.shuffle(temp_right_fi)
        right_feature_indices = temp_right_fi [:self.featureNumber]
        
        temp_left_fi = np.arange(self.x.shape[1])
        np.random.shuffle(temp_left_fi)
        left_feature_indices = temp_left_fi [:self.featureNumber]
                    
        x = self.x[self.data_indices,self.selected_feature]
        left_list= []
        right_list= []
        for k in range(x.shape[0]):
            if x[k]<=self.decision_value:
                left_list.append(k) 
            else:
                right_list.append(k)   
        left_tree_indices = np.asarray(left_list)
        righ_tree_indices = np.asarray(right_list)
                
#        if left_tree_indices.shape[0] < 2 or righ_tree_indices.shape[0] < 2:
#            print("number of samples for tree is less than 2")

        self.left_tree = RandomTree(self.x, self.y, self.featureNumber, left_feature_indices, self.data_indices[left_tree_indices], max_depth=self.max_depth-1, min_node_sample=self.min_node_sample)
        self.right_tree = RandomTree(self.x, self.y, self.featureNumber,  right_feature_indices, self.data_indices[righ_tree_indices], max_depth=self.max_depth-1, min_node_sample=self.min_node_sample)
          
    def find_estimate (self, x):
        return np.array([self.find_one_estimate(xi) for xi in x])

    def find_one_estimate(self, xi):
        if self.weighted_sd== float('inf') or self.max_depth <= 0: 
            return self.avg_price
        if xi[self.selected_feature]<=self.decision_value:
             smaller_tree = self.left_tree
        else:
            smaller_tree = self.right_tree
        return smaller_tree.find_one_estimate(xi)   
        

tree_number = 1;
feature_number = 3;
forest_depth = float('inf')
min_leaf_sample = 1;


avg_errors = np.zeros((2,10))
error_values_train = np.zeros((5,10))
error_values_test  = np.zeros((5,10)) 

tree_values = ([1,3,5,8,10,15])
        
K = 5
#validation
for l in range (len(tree_values)):
    feature_number = tree_values[l]
    print(tree_number)
    for valInd in range(K):
        forest2 = [];
        print('Fold K =', valInd)
        valLength = len(valData)
        valParts = np.arange(valLength)
        testIndex =  list(range(int(valLength/K) * valInd , int(valLength/K) * (valInd + 1)))
        test_x = valData[testIndex]
        price_of_test = valPrc[ testIndex]
        data_train = valData[ np.delete(valParts, testIndex )]
        price_train = valPrc[ np.delete(valParts, testIndex )]        
        initial_indices=np.array(range(data_train.shape[0]))
                
        for i in range(tree_number):
        #    temp_di = np.arange(data_train.shape[0])
        #    np.random.shuffle(temp_di) 
        #    x_indices = temp_di [:sample_size]    
            
            x_indices =  np.random.randint(data_train.shape[0], size=data_train.shape[0])
            temp_fi = np.arange(data_train.shape[1])
            np.random.shuffle(temp_fi)
            feature_indices = temp_fi [:feature_number]        
            
            forest2.append(RandomTree(data_train[x_indices], price_train[x_indices], feature_number,  feature_indices, 
                       data_indices=initial_indices, max_depth = forest_depth, min_node_sample = min_leaf_sample))
            
        print("prediction")
        predict_outcome =  np.mean([t.find_estimate(data_train) for t in forest2], axis=0)  
        predict_outcome_test =  np.mean([t.find_estimate(test_x) for t in forest2], axis=0)
        
        prediction_mat = np.asmatrix(predict_outcome);
        prediction_mat = prediction_mat.T
        
        nan_count = 0
        sum_er = 0
        real_price = (np.max(price_train)-np.min(price_train))*price_train+np.min(price_train)
        real_prediction = (np.max(price_train)-np.min(price_train))*prediction_mat+np.min(price_train)
        
        for i in range (1, prediction_mat.shape[0]-1):
            if not (np.isnan(prediction_mat[i,0])):
                sum_er = sum_er + np.absolute(real_prediction[i,0] - real_price[i,0])/real_price[i,0]
            else:
                nan_count += 1
        
        error_values_train[valInd,l] = sum_er / (prediction_mat.shape[0] - nan_count); 
        
        
        prediction_mat2 = np.asmatrix(predict_outcome_test);
        prediction_mat2 = prediction_mat2.T
        
        nan_count2 = 0
        sum_er2 = 0
        real_price2 = (np.max(price_of_test)-np.min(price_of_test))*price_of_test+np.min(price_of_test)
        real_prediction2 = (np.max(price_of_test)-np.min(price_of_test))*prediction_mat2+np.min(price_of_test)
        
        for i in range (1, prediction_mat2.shape[0]-1):
            if not (np.isnan(prediction_mat2[i,0])):
                sum_er2 = sum_er2 + np.absolute(real_prediction2[i,0] - real_price2[i,0])/real_price2[i,0]
            else:
                nan_count2 += 1
        
        error_values_test[valInd,l] = sum_er2 / (prediction_mat2.shape[0] - nan_count2); 
            
    avg_errors [0,l] = np.mean(error_values_train[:,l])
    avg_errors [1,l] = np.mean(error_values_test[:,l])

print(avg_errors)

