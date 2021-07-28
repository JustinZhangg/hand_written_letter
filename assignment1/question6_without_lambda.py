import numpy as np;
import numpy.matlib 
import math
import matplotlib.pyplot as plt
import csv
from scipy.io import loadmat

#     Load the EMNIST dataset:
#     Reading the data file gives an array of img_size by no. of samples
#     so we will work on the transposed array instead 

mnist = loadmat('emnist-letters-1k.mat')
# Read the train set
x_train = mnist['train_images']
# Read the train labels
trainlabels = mnist['train_labels']
# Read the test set
x_test = mnist['test_images']
# Read the test labels
testlabels = mnist['test_labels']
# normalise data
normalised_train = np.array(x_train)/255
normalised_test = np.array(x_test)/255
# randomise data with corresponding label
assert len(normalised_train) == len(trainlabels)
p = numpy.random.permutation(len(normalised_train))
randomised_train =  normalised_train[p]
randomised_trainlabels = trainlabels[p]

assert len(normalised_test) == len(testlabels)
p1 = numpy.random.permutation(len(normalised_test))
randomised_test =  normalised_test[p1]
randomised_testlabels = testlabels[p1]
# split train data into dev set 8:2
train_dev_percent = int(0.2 * randomised_train.shape[0])
randomised_dev = randomised_train[:train_dev_percent]
randomised_dev_labels = randomised_trainlabels[:train_dev_percent]
randomised_train_set = randomised_train[train_dev_percent:]
randomised_train_set_labels = randomised_trainlabels[train_dev_percent:]

n_samples, img_size = randomised_train_set.shape

# The EMNIST contains A-Z so we will set the number of labels as 26
nlabels = 26

# transfer labels into array, index of eliment equal to 1 is the label of that sample
y_train = np.zeros((randomised_train_set_labels.shape[0], nlabels))
y_test  = np.zeros((randomised_dev_labels.shape[0], nlabels))

for i in range(0,randomised_train_set_labels.shape[0]):   
    y_train[i, randomised_train_set_labels[i].astype(int)]=1
    
for i in range(0,randomised_dev_labels.shape[0]):    
    y_test[i, randomised_dev_labels[i].astype(int)]=1
    
# The number of epochs is a hyperparameter that defines the number times that the learning algorithm 
# will work through the entire training dataset.

# The batch size is a hyperparameter that defines the number of samples to work through before 
# updating the internal model parameters. 

# ref: https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/

n_epoch = 250
batch_size = 50#100
n_batches = int(math.ceil(n_samples/batch_size))

# define the size of each of the layers in the network
n_input_layer  = img_size
n_hidden_layer = 50 # number of neurons of the hidden layer. 0 deletes this layer
n_output_layer = nlabels

# Add another hidden layer
n_hidden_layer2 = 0#100 # number of neurons of the hidden layer. 0 deletes this layer

# eta is the learning rate
eta = 0.05

# set lambda value

lambdalist = [0.000001, 0.000003, 0.00001, 0.00003, 0.0001]
# how many runs
different_run = 5
# record validation error
validation_error = np.zeros((len(lambdalist), different_run))

# for question6
for values in range(0,len(lambdalist)):
    for num_tries in range(0,different_run):        
        # Initialize a simple network
        # For W1 and W2 columns are the input and the rows are the output.
        # W1: Number of columns (input) needs to be equal to the number of features 
        #     of the  MNIST digits, thus p. Number of rows (output) should be equal 
        #     to the number of neurons of the hidden layer thus n_hidden_layer.
        # W2: Number of columns (input) needs to be equal to the number of neurons 
        #     of the hidden layer. Number of rows (output) should be equal to the 
        #     number of digits we wish to find (classification).
        
        Xavier_init=True
        
        if Xavier_init:   
            if n_hidden_layer>0:
                W1 = np.random.randn(n_hidden_layer, n_input_layer) * np.sqrt(1 / (n_input_layer))
                if n_hidden_layer2>0:
                    W2 = np.random.randn(n_hidden_layer2, n_hidden_layer) * np.sqrt(1 / (n_hidden_layer))
                    W3 = np.random.randn(n_output_layer, n_hidden_layer2) * np.sqrt(1 / (n_hidden_layer2))
                else:
                    W2 = np.random.randn(n_output_layer, n_hidden_layer) * np.sqrt(1 / (n_hidden_layer))
            else:
                W1 = np.random.randn(n_output_layer, n_input_layer) * np.sqrt(1 / (n_input_layer))
        else:
            if n_hidden_layer>0: 
                W1 = np.random.uniform(0,1,(n_hidden_layer, n_input_layer))
                W2 = np.random.uniform(0,1,(n_output_layer, n_hidden_layer))
        
                # The following normalises the random weights so that the sum of each row =1
                W1 = np.divide(W1,np.matlib.repmat(np.sum(W1,1)[:,None],1,n_input_layer))
                W2 = np.divide(W2,np.matlib.repmat(np.sum(W2,1)[:,None],1,n_hidden_layer))
        
                if n_hidden_layer2>0:
                    W3=np.random.uniform(0,1,(n_output_layer,n_hidden_layer2))
                    W3=np.divide(W3,np.matlib.repmat(np.sum(W3,1)[:,None],1,n_hidden_layer2))
        
                    W2=np.random.uniform(0,1,(n_hidden_layer2,n_hidden_layer))
                    W2=np.divide(W2,np.matlib.repmat(np.sum(W2,1)[:,None],1,n_hidden_layer))
            else:
               W1 = np.random.uniform(0,1,(n_output_layer, n_input_layer))
               W1 = np.divide(W1,np.matlib.repmat(np.sum(W1,1)[:,None],1,n_input_layer))
               
        # Initialize the biases
        bias_W1 = np.zeros((n_output_layer,))
        if n_hidden_layer>0: 
            bias_W1 = np.zeros((n_hidden_layer,))
            bias_W2 = np.zeros((n_output_layer,))
            if n_hidden_layer2>0:    
                bias_W3=np.zeros((n_output_layer,))
                bias_W2=np.zeros((n_hidden_layer2,))
            
        
        # Train the network       
        for i in range(0,n_epoch):
            
            # Initialise the gradients for each batch
            dW1 = np.zeros(W1.shape)
            if n_hidden_layer>0: 
                dW2 = np.zeros(W2.shape)
            
            # We will shuffle the order of the samples each epoch
            shuffled_idxs = np.random.permutation(n_samples)
            shuffled_data_set = randomised_train_set[shuffled_idxs]
            shuffled_label_set = y_train[shuffled_idxs]
            for batch in range(0,n_batches):
                # Initialise the gradients for each batch
                dW1 = np.zeros(W1.shape)
                dbias_W1 = np.zeros(bias_W1.shape)
                if n_hidden_layer>0: 
                    dW2 = np.zeros(W2.shape)
                    dbias_W2 = np.zeros(bias_W2.shape)
                    if n_hidden_layer2 > 0:
                        dW3 = np.zeros(W3.shape)
                        dbias_W3 = np.zeros(bias_W3.shape)
                
                # shuffled input
                #idx = shuffled_idxs[batch*batch_size]
                
                x0 = shuffled_data_set[batch*batch_size:batch*batch_size+batch_size]
               
                # Form the desired output, the correct neuron should have 1 the rest 0
                
                desired_output = shuffled_label_set[batch*batch_size:batch*batch_size+batch_size]

                h1 = np.matmul(W1,x0.T)+np.tile(bias_W1,(batch_size,1)).T
                # Apply the RELU function
                x1 = np.maximum(h1,0)
                
                if n_hidden_layer > 0:
                    # Neural activation: hidden layer -> output layer
                    h2 = np.matmul(W2,x1)+np.tile(bias_W2,(batch_size,1)).T

                    # Apply the RELU function
                    x2 = np.maximum(h2,0)
                  
                    if n_hidden_layer2 > 0:
                        # Neural activation: hidden layer 1 -> hidden layer 2
                        h3 = np.dot(W3,x2)+np.tile(bias_W3,(batch_size,1)).T

                        # Apply the RELU function
                        x3 = np.maximum(h3,0)
                                    
                        # Compute the error signal
                        e_n = desired_output - x3.T
                                    
                        # Backpropagation: output layer -> hidden layer 2
                        delta3 = e_n.T * np.where(x3 <= 0, 0, 1)
                                    
                        dW3 += np.matmul(delta3,x2.T)
                        dbias_W3 += np.sum(delta3)
                                    
                        # Backpropagation: hidden layer -> input layer
                        delta2 = np.where(x2 <= 0, 0, 1) * np.matmul(W3.T, delta3)
                    else:
                        # Compute the error signal
                        e_n = desired_output - x2.T
                        # Backpropagation: output layer -> hidden layer
                        delta2 = e_n.T * np.where(x2 <= 0, 0, 1)
                   
                    dW2 += np.matmul(delta2, x1.T) 
                                   
                    dbias_W2 = np.sum(delta2)

                    # Backpropagation: hidden layer -> input layer
                    delta1 = np.where(x1 <= 0, 0, 1) * np.dot(W2.T, delta2)
                else:
                                
                    e_n = np.subtract(desired_output,x1.T)
                                
                    delta1 = np.where(x1 <= 0, 0, 1) * e_n
                                          
                dW1 += np.matmul(delta1,x0) 

                dbias_W1 += np.sum(delta1)        
                        
                # After each batch update the weights using accumulated gradients
                # as we can't let 0 undefined, so simply set when x = 0, sign(x)=1 
                #####################
                # this part code is for no L1    
                W1 += eta*dW1/batch_size
                bias_W1 += eta*dbias_W1/batch_size              
                if n_hidden_layer > 0:
                    W2 += eta*dW2/batch_size 
                    bias_W2 += eta*dbias_W2/batch_size
                    if n_hidden_layer2 > 0:
                        W3 += eta*dW3/batch_size 
                        bias_W3 += eta*dbias_W3/batch_size
                #######################
                
                #W1 += eta*dW1/batch_size - eta * lambdalist[values] * np.where(W1 < 0, -1, 1)
                #bias_W1 += eta*dbias_W1/batch_size
                     
                #if n_hidden_layer > 0:
                #    W2 += eta*dW2/batch_size - eta * lambdalist[values] * np.where(W2 < 0, -1, 1)
                #    bias_W2 += eta*dbias_W2/batch_size
                #    if n_hidden_layer2 > 0:
                #        W3 += eta*dW3/batch_size - eta * lambdalist[values] * np.where(W3 < 0, -1, 1)
                #        bias_W3 += eta*dbias_W3/batch_size
                    
        # TODO: use the test set to compute the network's accuracy
        n = randomised_dev.shape[0]
        # calculate validation error
        for datapoint in range(0,n):
            # Input (random element from the dataset)
                    x0 = randomised_dev[datapoint]
                    
                    # Form the desired output, the correct neuron should have 1 the rest 0
                    desired_output = y_test[datapoint]
        
                    # Neural activation: input layer -> hidden/output layer
                    h1 = np.dot(W1,x0)+bias_W1
        
                    # Apply the RELU function
                    x1 = np.maximum(h1, 0)
                    
                    if n_hidden_layer>0: 
                        # Neural activation: hidden layer -> hidden/output layer
                        h2 = np.dot(W2,x1)+bias_W2
        
                        # Apply the RELU function
                        x2 = np.maximum(h2, 0)
                    
                        if n_hidden_layer2 > 0:
                            # Neural activation: hidden layer 1 -> hidden layer 2
                            h3 = np.dot(W3,x2)+bias_W3
        
                            # Apply the RELU function
                            x3 = np.maximum(h3, 0)
                        
                            # Compute the error signal
                            e_n = desired_output - x3
                            validation_error[values][num_tries] = validation_error[values][num_tries] + 0.5*np.sum(np.square(e_n))/randomised_dev.shape[0] 
                        else:
                            # Compute the error signal
                            e_n = desired_output - x2
                            validation_error[values][num_tries] = validation_error[values][num_tries] + 0.5*np.sum(np.square(e_n))/randomised_dev.shape[0] 
                    else:
                        # Compute the error signal
                        e_n = desired_output - x1
                        validation_error[values][num_tries] = validation_error[values][num_tries] + 0.5*np.sum(np.square(e_n))/randomised_dev.shape[0] 
times = [1,2,3,4,5]
yerr = np.zeros(len(lambdalist))
mean_error = np.zeros(len(lambdalist))
for v in range(0,len(lambdalist)):
    mean_error[v] = np.mean(validation_error[v])
    yerr[v] = np.std(validation_error[v])
    
plt.xlabel('Runs')
plt.ylabel('Error')
plt.errorbar(times, mean_error, yerr=yerr)
plt.title('Average error against lambdas')
plt.show()
print(mean_error)
                    


