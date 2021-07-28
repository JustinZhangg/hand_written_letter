import numpy as np;
import numpy.matlib 
import math
import matplotlib.pyplot as plt
import csv
from scipy.io import loadmat
from sklearn.utils import shuffle

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
#normalised_train = np.array(x_train)/255
#normalised_test = np.array(x_test)/255

# normalize data by SD Normalization
#x_train = np.array([(samples - np.mean(samples))/np.std(samples) for samples in x_train])
#normalised_train = x_train
#x_test = np.array([(samples - np.mean(samples))/np.std(samples) for samples in x_test])
#normalised_test = x_test

# normalize data by min-max Normalization
x_train = np.array([(samples - min(samples))/(max(samples)-min(samples)) for samples in x_train])
normalised_train = x_train
x_test = np.array([(samples - min(samples))/(max(samples)-min(samples)) for samples in x_test])
normalised_test = x_test
# randomise data with corresponding label
randomised_train, randomised_trainlabels = shuffle(normalised_train, trainlabels, random_state=0)
randomised_test, randomised_testlabels = shuffle(normalised_test, testlabels, random_state=0)

# split train data into dev set 8:2
train_dev_percent = int(0.2 * randomised_train.shape[0])
randomised_dev = randomised_train[:train_dev_percent]
randomised_dev_labels = randomised_trainlabels[:train_dev_percent]
randomised_train_set = randomised_train[train_dev_percent:]
randomised_train_set_labels = randomised_trainlabels[train_dev_percent:]

n_samples, img_size = randomised_train.shape

# The EMNIST contains A-Z so we will set the number of labels as 26
nlabels = 26

# transfer labels into array, index of eliment equal to 1 is the label of that sample
y_train = np.zeros((randomised_trainlabels.shape[0], nlabels))
y_test  = np.zeros((randomised_testlabels.shape[0], nlabels))

for i in range(0,randomised_trainlabels.shape[0]):   
    y_train[i, randomised_trainlabels[i].astype(int)]=1
    
for i in range(0,randomised_testlabels.shape[0]):    
    y_test[i, randomised_testlabels[i].astype(int)]=1
    
# The number of epochs is a hyperparameter that defines the number times that the learning algorithm 
# will work through the entire training dataset.

# The batch size is a hyperparameter that defines the number of samples to work through before 
# updating the internal model parameters. 

# ref: https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/

n_epoch = 100
batch_size = 50#100
n_batches = int(math.ceil(n_samples/batch_size))

# define the size of each of the layers in the network
n_input_layer  = img_size
#n_hidden_layer = 50 # number of neurons of the hidden layer. 0 deletes this layer
n_output_layer = nlabels

# Add another hidden layer
n_hidden_layer2 = 0#100 # number of neurons of the hidden layer. 0 deletes this layer

# eta is the learning rate
eta = 0.05

# set lambda value
lambdalist = [0.000001, 0.000003, 0.00001, 0.00003, 0.0001]
lamda = 0.00003
# how many runs
different_run = 5
# record validation error
validation_error = np.zeros((len(lambdalist), different_run))

# number of neuron
num_neuron = [50, 100, 200, 400]
# record accuracy
accuracy_list = np.zeros((len(num_neuron), different_run))

# for question6
for num_neurons in range(0,len(num_neuron)):
    for num_tries in range(0,different_run):        
        # Initialize a simple network
        # For W1 and W2 columns are the input and the rows are the output.
        # W1: Number of columns (input) needs to be equal to the number of features 
        #     of the  MNIST digits, thus p. Number of rows (output) should be equal 
        #     to the number of neurons of the hidden layer thus n_hidden_layer.
        # W2: Number of columns (input) needs to be equal to the number of neurons 
        #     of the hidden layer. Number of rows (output) should be equal to the 
        #     number of digits we wish to find (classification).
        
        n_hidden_layer = num_neuron[num_neurons]
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
            shuffled_data_set = randomised_train[shuffled_idxs]
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
                #W1 += eta*dW1/batch_size
                #bias_W1 += eta*dbias_W1/batch_size              
                #if n_hidden_layer > 0:
                #    W2 += eta*dW2/batch_size 
                #    bias_W2 += eta*dbias_W2/batch_size
                #    if n_hidden_layer2 > 0:
                #        W3 += eta*dW3/batch_size 
                #        bias_W3 += eta*dbias_W3/batch_size
                #######################
                
                W1 += eta*dW1/batch_size - eta * lamda * np.where(W1 < 0, -1, 1)
                bias_W1 += eta*dbias_W1/batch_size
                     
                if n_hidden_layer > 0:
                    W2 += eta*dW2/batch_size - eta * lamda * np.where(W2 < 0, -1, 1)
                    bias_W2 += eta*dbias_W2/batch_size
                    if n_hidden_layer2 > 0:
                        W3 += eta*dW3/batch_size - eta * lamda * np.where(W3 < 0, -1, 1)
                        bias_W3 += eta*dbias_W3/batch_size
                    

        # TODO: use the test set to compute the network's accuracy
        n = randomised_test.shape[0]
        
        p_ra = 0
        correct_value = np.zeros((n,))
        predicted_value = np.zeros((n,))
        
        for num in range(0, n):
            x0 = randomised_test[num]
            y = y_test[num]
            
            correct_value[num] = np.argmax(y)
            
            h1 = np.dot(W1, x0) + bias_W1    
            x1 = np.maximum(h1, 0)
            if n_hidden_layer > 0:
                h2 = np.dot(W2, x1) + bias_W2
                x2 = np.maximum(h2, 0)
                if n_hidden_layer2 > 0:     
                    h3 = np.dot(W3, x2) + bias_W3    
                    x3 = np.maximum(h3, 0)
        
                    predicted_value[num] = np.argmax(x3)
                else:            
                    predicted_value[num] = np.argmax(x2)
            else:            
                predicted_value[num] = np.argmax(x1) 
                    
            if predicted_value[num] == correct_value[num]: 
                p_ra = (p_ra + 1)
        
        accuracy = 100*p_ra/n 
        print("number of neuron = ",num_neuron[num_neurons],"number of tries = ",num_tries,"Accuracy = ", accuracy)
        accuracy_list[num_neurons][num_tries] = accuracy
yerr = np.zeros(len(num_neuron))
mean_error = np.zeros(len(num_neuron))

for v in range(0,len(num_neuron)):
    mean_error[v] = np.mean(accuracy_list[v])
    yerr[v] = np.std(accuracy_list[v])

plt.xlabel('number of neurons')
plt.ylabel('Accuracy')
plt.errorbar(num_neuron, mean_error, yerr=yerr)
plt.title('Average accuracy against number of neurons')
plt.show()
                    


