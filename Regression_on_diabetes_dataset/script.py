import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    _, d = X.shape

    label = np.unique(y)
    classes_count = len(label)

    means = np.zeros((d, classes_count))

    for i, label in enumerate(label):
        class_data = X[y.flatten() == label]
        class_mean = np.mean(class_data, axis=0)  # Calculating the mean for the current class
        for j in range(len(class_mean)):         # Iterating over features
            means[j, i] = class_mean[j] 

    covmat =  np.cov(X.T, bias=True)
    return means, covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    label = np.unique(y)
    classes_count = len(label)
    _, d = X.shape

    means = [[0.0 for _ in range(classes_count)] for _ in range(d)]
    means = np.array(means) 
    covmats = []

    for i, label in enumerate(label):
        # Filtering data for class
        class_data = X[y.flatten() == label]
        
        # We are Geting mean for each class
        mean_vector = np.mean(class_data, axis=0)
        for j in range(mean_vector.shape[0]):
            means[j, i] = mean_vector[j]

        
        # We are Calculating  covariance matrix for each class here
        matrix = np.cov(class_data, rowvar=False)
        covmats.append(matrix)
    
    return means, covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    N, _ = Xtest.shape  

    num_of_classes = means.shape[1]
    inv_cov = inv(covmat)    
    ypred = np.zeros((N,1))

    for i in range(N):  
        x = Xtest[i, :]  
        discriminants = np.zeros(num_of_classes)  

        for class_index in range(num_of_classes):
            mean = means[:, class_index]  
            diff = x - mean  
            temp = np.dot(inv_cov, diff)  
            quadratic_form = np.dot(diff.T, temp)  
            discriminants[class_index] = -0.5 * quadratic_form  


        ypred[i] = np.argmax(discriminants) + 1  

    acc = np.mean(ypred.flatten() == ytest.flatten())
    return acc, ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    N, _ = Xtest.shape
    ypred = np.zeros((N, 1))

    def get_likelihoods():
        classes_count = len(covmats)
        likelihoods = np.zeros((N, classes_count))
        
        for index, cov_matrix in enumerate(covmats):
            mean = means[:, index]

            # Get determinant and the inverse of covariance
            det_cov = det(cov_matrix)
            inv_cov = inv(cov_matrix)

            # We are calculating the likelihood
            for i in range(N):
                diff = Xtest[i, :] - mean
                diff = diff.reshape((-1,1))
                exponent = np.sum(-0.5 * np.dot(diff.T, np.dot(inv_cov, diff)))
                normalization_factor = 1 / np.sqrt((2 * np.pi) ** len(mean) * det_cov)
                likelihood = normalization_factor * np.exp(exponent)
                likelihoods[i, index] = likelihood
        
        return likelihoods

    likelihoods = get_likelihoods()

    # We will select the class with the maximum likelihood
    ypred = np.argmax(likelihoods, axis=1)

    # We update the labels predicted from 0 index to correct values
    ypred = ypred + np.ones(ypred.shape)

    # Finally, we are calculating the accuracy
    accuracy = np.mean(ypred.flatten() == ytest.flatten())
    return accuracy, ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD
    w = np.matmul(np.matmul(inv(np.matmul(X.T,X)), X.T),y)                                    
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    _, d = X.shape
    ridge_term = lambd*np.identity(d)
    w = np.matmul(np.matmul(inv(ridge_term + np.matmul(X.T,X)), X.T),y)                                                                                    
    return w

def testOLERegression(w, Xtest, ytest):
    """
    Test the Ordinary Least Squares Regression model.

    Parameters:
    w : numpy.ndarray
        The weight vector of shape (d, 1).
    Xtest : numpy.ndarray
        Test feature matrix of shape (N, d).
    ytest : numpy.ndarray
        True target values of shape (N, 1).

    Returns:
    float
        Mean Squared Error (MSE).
    """
    # Number of samples
    N = Xtest.shape[0]
    
    # Predicted values
    ypred = Xtest @ w  # i am Using @ for matrix multiplication

    # I am calculating Mean Squared Error
    residuals = ytest - ypred
    squared_residuals = residuals ** 2
    mse = squared_residuals.mean()  # Directly using mean for efficiency

    return mse


def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD
    w = w.reshape(w.shape[0], 1)
    residual = y - np.matmul(X, w)
    # I am claculating  the dot products separately and after this i will combine them 
    residual_term = np.dot(residual.T, residual) * 0.5
    reg_term = np.dot(w.T, w) * (lambd * 0.5)
    error = residual_term + reg_term

    error = error[0][0]
    error_grad = -1 * np.matmul(X.T, residual) + lambd * w

    error_grad = error_grad.flatten()                              
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD

    # Get the number of samples
    N = len(x)

    # Reshape x into a column vector
    x = x[:, np.newaxis]

    # Initialize the polynomial features matrix
    Xp = np.array([x.flatten()**i for i in range(p + 1)]).T

    return Xp


# Main script
if __name__ == "__main__":
    # Problem 1
    # load the sample data                                                                 
    if sys.version_info.major == 2:
        X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
    else:
        X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

    # LDA
    means,covmat = ldaLearn(X,y)
    ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
    print('LDA Accuracy = '+str(ldaacc))
    # QDA
    means,covmats = qdaLearn(X,y)
    qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
    print('QDA Accuracy = '+str(qdaacc))

    # plotting boundaries
    x1 = np.linspace(-5,20,100)
    x2 = np.linspace(-5,20,100)
    xx1,xx2 = np.meshgrid(x1,x2)
    xx = np.zeros((x1.shape[0]*x2.shape[0],2))
    xx[:,0] = xx1.ravel()
    xx[:,1] = xx2.ravel()

    fig = plt.figure(figsize=[12,6])
    plt.subplot(1, 2, 1)

    zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
    plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
    plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
    plt.title('LDA')

    plt.subplot(1, 2, 2)

    zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
    plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
    plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
    plt.title('QDA')

    plt.show()
    # Problem 2
    if sys.version_info.major == 2:
        X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
    else:
        X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

    # add intercept
    X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
    Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

    w = learnOLERegression(X,y)
    mle = testOLERegression(w,Xtest,ytest)

    w_i = learnOLERegression(X_i,y)
    mle_i = testOLERegression(w_i,Xtest_i,ytest)

    print('MSE without intercept '+str(mle))
    print('MSE with intercept '+str(mle_i))

    # Problem 3
    k = 101
    lambdas = np.linspace(0, 1, num=k)
    i = 0
    mses3_train = np.zeros((k,1))
    mses3 = np.zeros((k,1))
    for lambd in lambdas:
        w_l = learnRidgeRegression(X_i,y,lambd)
        if lambd==0.06:
            w_r = w_l
        mses3_train[i] = testOLERegression(w_l,X_i,y)
        mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
        i = i + 1
    fig = plt.figure(figsize=[12,6])
    plt.subplot(1, 2, 1)
    plt.plot(lambdas,mses3_train)
    plt.title('MSE for Train Data')
    plt.subplot(1, 2, 2)
    plt.plot(lambdas,mses3)
    plt.title('MSE for Test Data')

    fig = plt.figure(figsize=[12,6])
    plt.subplot(1, 2, 1)
    plt.plot(range(X_i.shape[1]),w_i)
    plt.title('Weights for OLE')
    plt.subplot(1, 2, 2)
    plt.plot(range(X_i.shape[1]),w_r)
    plt.title('Weights for Ridge')

    plt.show()
    # Problem 4
    k = 101
    lambdas = np.linspace(0, 1, num=k)
    i = 0
    mses4_train = np.zeros((k,1))
    mses4 = np.zeros((k,1))
    opts = {'maxiter' : 20}    # Preferred value.                                                
    w_init = np.ones((X_i.shape[1],1)).flatten()
    for lambd in lambdas:
        args = (X_i, y, lambd)
        w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
        w_l = np.transpose(np.array(w_l.x))
        w_l = np.reshape(w_l,[len(w_l),1])
        mses4_train[i] = testOLERegression(w_l,X_i,y)
        mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
        i = i + 1
    fig = plt.figure(figsize=[12,6])
    plt.subplot(1, 2, 1)
    plt.plot(lambdas,mses4_train)
    plt.plot(lambdas,mses3_train)
    plt.title('MSE for Train Data')
    plt.legend(['Using scipy.minimize','Direct minimization'])

    plt.subplot(1, 2, 2)
    plt.plot(lambdas,mses4)
    plt.plot(lambdas,mses3)
    plt.title('MSE for Test Data')
    plt.legend(['Using scipy.minimize','Direct minimization'])
    plt.show()


    # Problem 5
    pmax = 7
    lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
    mses5_train = np.zeros((pmax,2))
    mses5 = np.zeros((pmax,2))
    for p in range(pmax):
        Xd = mapNonLinear(X[:,2],p)
        Xdtest = mapNonLinear(Xtest[:,2],p)
        w_d1 = learnRidgeRegression(Xd,y,0)
        mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
        mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
        w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
        mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
        mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

    fig = plt.figure(figsize=[12,6])
    plt.subplot(1, 2, 1)
    plt.plot(range(pmax),mses5_train)
    plt.title('MSE for Train Data')
    plt.legend(('No Regularization','Regularization'))
    plt.subplot(1, 2, 2)
    plt.plot(range(pmax),mses5)
    plt.title('MSE for Test Data')
    plt.legend(('No Regularization','Regularization'))
    plt.show()
