# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:41:07 2024

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

links=r'C:\Users\User\OneDrive\Desktop\Fordham\Data Mining (CISC-5790-L02)\HW1_dataset'

#Split 'train-1000-100.csv' into 3 smaller files
    
def split_train_1000():
    train_1000_data = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\Fordham\Data Mining (CISC-5790-L02)\HW1_dataset\train-1000-100.csv")
    
    train_50_data=train_1000_data.iloc[:50]
    train_100_data=train_1000_data.iloc[:100]
    train_150_data=train_1000_data.iloc[:150]
    
    train_50_data.to_csv(f'{links}\\train-50(1000)-100.csv', index=False)# index = false to make sure not taking the index of dataset into a new csv file
    train_100_data.to_csv(f'{links}\\train-100(1000)-100.csv', index=False)
    train_150_data.to_csv(f'{links}\\train-150(1000)-100.csv', index=False)
split_train_1000() #run function

datasets={
    'dataset 1 (100-10)': ('train-100-10.csv', 'test-100-10.csv'),
    'dataset 2 (100-100)': ('train-100-100.csv', 'test-100-100.csv'),
    'dataset 3 (1000-100)': ('train-1000-100.csv', 'test-1000-100.csv'),
    'dataset 4 [50(1000)-100]': ('train-50(1000)-100.csv', 'test-1000-100.csv'),
    'dataset 5 [100(1000)-100]': ('train-100(1000)-100.csv', 'test-1000-100.csv'),
    'dataset 6 [150(1000)-100]': ('train-150(1000)-100.csv', 'test-1000-100.csv')
    }
def preprocesssing_data(train_file, test_file):
    # Split data into features X and target y
    train_data = pd.read_csv(f'{links}\\{train_file}') #somehow it was str, which i dont understand
    test_data =pd.read_csv(f'{links}\\{test_file}') #somehow it was str, which i dont understand
    
    #DEBUGGING STEP AFTER TRYING TO EXECUTE THE CODE AND GET THE ISSUES of can't multipy strings
    
    # Check for missing or non-numeric values in the training set
    print("Missing values in training data:", train_data.isnull().sum())
    print("Unique values in training data:", train_data.apply(lambda x: x.unique()))
    
    #found out that all values in training data become unique values and there were 2 extra columns 'unnamed 11' and unnamed 12'
    # dropping the unnamed columns
    # Using errors='ignore' allows the drop function to proceed without raising an error if the specified columns don't exist because idk why there are that 2 columns
    train_data.drop(columns=['Unnamed: 11', 'Unnamed: 12'], errors='ignore', inplace=True)
    test_data.drop(columns=['Unnamed: 11', 'Unnamed: 12'], errors='ignore', inplace=True)
    
    #using .values to convert pandas DataFrames to numpy arrays before performing matrix operations
    # Convert to float
    #iloc(rows, columns)
    x_train = train_data.iloc[:, :-1].astype(float).values #
    y_train = train_data.iloc[:, -1].astype(float).values
    x_test = test_data.iloc[:, :-1].astype(float).values
    y_test = test_data.iloc[:, -1].astype(float).values

    print("Type of x_train:", type(x_train))
    print("Type of y_train:", type(y_train))

    return x_train, y_train, x_test, y_test

# Ensure shapes match

def L2_ridge_regression(x,y,lambd):
    n=x.shape[1] #shape[] refer to numbers of columns (1)/rows(0) in matrix x
    I=np.eye(n) #creates an identity matrix of size n x n
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("I shape:", I.shape)
    # closed-form solution for L2
    w=np.linalg.inv(x.T@x+lambd*I)@x.T@y #one weight vector for the whole dataset of X features
    # use @ to multiply matrix
    return w

def MSE(x,y,w):
    y_predictions = x @ w
    mse = np.mean((y_predictions - y) ** 2)
    return mse

global_lambdas_list = []
global_mse_train_list = []
global_mse_test_list = []
def model_evaluation(x_train,y_train,x_test,y_test,lambdas_range):
    lambdas_list=[]
    mse_train_list=[]
    mse_test_list=[]
    for lambd in lambdas_range:
        w=L2_ridge_regression(x_train,y_train,lambd)
        
        mse_train = MSE(x_train, y_train, w)
        mse_test = MSE(x_test, y_test, w)
    
        lambdas_list.append(lambd)
        mse_train_list.append(mse_train)
        mse_test_list.append(mse_test)
        
        # Print MSE values
        print(f'Lambda: {lambd}, Training MSE: {mse_train}, Testing MSE: {mse_test}')
     # Store results in global lists
    global global_lambdas_list, global_mse_train_list, global_mse_test_list
    global_lambdas_list = lambdas_list
    global_mse_train_list = mse_train_list
    global_mse_test_list = mse_test_list
    return lambdas_list, mse_test_list,mse_train_list
    
def plot_mse_lambd(datasets, lambdas_range):

    for dataset, (train_file, test_file) in datasets.items():
        x_train, y_train, x_test, y_test=preprocesssing_data(train_file, test_file)
        
        lambdas_list, mse_test_list,mse_train_list=model_evaluation(x_train, y_train, x_test, y_test, lambdas_range)
        
        min_test_mse=min(mse_test_list)
        best_lambda = lambdas_list[mse_test_list.index(min_test_mse)]
        
    
        plt.plot(lambdas_list, mse_train_list, label=f'Training MSE - {dataset}', marker='o', color='blue')
        plt.plot(lambdas_list, mse_test_list, label=f'Testing MSE - {dataset}', marker='x', color='red')
        plt.plot(best_lambda, min_test_mse, 'bo') #bo is Blue circle
        plt.annotate(f'min_test_mse: {min_test_mse:.2f}\nbest_lambda: {best_lambda}', xy = (best_lambda,min_test_mse))
        plt.xlabel('Lambda Values')
        plt.ylabel('MSE')
        plt.title(f'MSE vs. Lambda {lambdas_range} for Training and Testing Data')
        plt.grid(True)
        plt.legend()
        plt.show()
plot_mse_lambd(datasets, range(0,150))

dataset_2_4_5 = {
    'dataset 2 (100-100)': datasets['dataset 2 (100-100)'],
    'dataset 4 [50(1000)-100]' : datasets['dataset 4 [50(1000)-100]'],
    'dataset 5 [100(1000)-100]': datasets['dataset 5 [100(1000)-100]']
    }

plot_mse_lambd(dataset_2_4_5, range(1,151))


#there is n rows of y -> n data point/samples
#shuffle them up, then split them into 10 folds -> avoide bias 
#n data point is divided by numbers of folds we want, in this case is 10
global_avg_mse_per_lambd_list=[]
def cross_validation (y_train,x_train,lambdas_range,n_folds=10):
    y_train_rows=len(y_train)
    y_train_rows_index=np.arange(y_train_rows)
    fold_size=y_train_rows//n_folds #number of data in a fold
    #Creating 9 folds and handling the last indices into the 10th fold
    avg_mse_per_lambd_list=[]
    for lambd in lambdas_range:
        mse_fold_list=[]
        for i in range (n_folds-1): #range 0->8 in the dataset is 1->9
            folds=[y_train_rows_index[i*fold_size:(i+1)*fold_size]]
            folds.append(y_train_rows_index[(n_folds-1)*fold_size:]) #the rest data into the last fold, and append onto folder list
            validation_fold=folds[-1] #extract the last fold -> validation folder
            train_fold_index = np.setdiff1d(y_train_rows_index, validation_fold)# combine all the training set into a set
            
            # define training and validation dataset
            x_train_fold, y_train_fold = x_train[train_fold_index], y_train[train_fold_index]
            x_validation_fold, y_validation_fold=x_train[validation_fold], y_train[validation_fold]
            
            #Train the model with the current lambda
            w=L2_ridge_regression(x_train_fold,y_train_fold,lambd)
            
            #test on validation set and get the MSE
            mse_validation=MSE(x_validation_fold, y_validation_fold,w)
            mse_fold_list.append(mse_validation)
        avg_mse=np.mean(mse_fold_list)
        avg_mse_per_lambd_list.append(avg_mse)
        print(f'Lambda:{lambd}, Average validation MSE:{avg_mse}')
    global global_avg_mse_per_lambd_list
    global_avg_mse_per_lambd_list=avg_mse_per_lambd_list
    
    min_avg_mse=min(avg_mse_per_lambd_list)
    best_lambda=lambdas_range[avg_mse_per_lambd_list.index(min_avg_mse)]
    print(f'Best lambda from CV: {best_lambda}, Minimum average validation MSE: {min_avg_mse}')
    return best_lambda, min_avg_mse, avg_mse_per_lambd_list

def CV_execution(datasets,lambdas_range):
    for dataset, (train_file, test_file) in datasets.items():
        x_train, y_train, x_test, y_test=preprocesssing_data(train_file, test_file)
        best_lambda, min_avg_mse, avg_mse_per_lambd_list=cross_validation(y_train, x_train, lambdas_range)
        
        # Plot Cross-Validation MSE
        plt.plot(lambdas_range, avg_mse_per_lambd_list, label='Average CV MSE', linestyle='--', color='green')

        # Mark the best lambda from CV
        plt.plot(best_lambda, min_avg_mse, 'bo')  # blue circle
        plt.annotate(f'Min Avg MSE: {min_avg_mse:.2f}\nBest Lambda: {best_lambda}', 
                     xy=(best_lambda, min_avg_mse))
        
        plt.xlabel('Lambda Values')
        plt.ylabel('MSE')
        plt.title(f'MSE vs. Lambda for {dataset}')
        plt.grid(True)
        plt.legend()
        plt.show()

CV_execution(datasets, range(0, 151))
        
            
