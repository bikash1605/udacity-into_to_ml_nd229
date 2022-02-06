import numpy as np
np.random.seed(42)
#import matplotlib.pyplot as plt



# 1. Write the Step function
# 2. Calculate Y-predict
# 3. Calculate Preceptron Step

def stepFunction(t):
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])


# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X,y, W,b, learn_rate=0.01):
    for i in range(len(X)):
        y_hat = prediction(X[0],W,b) # Can be 1 or 0
        if(y_hat - y[i] == 1):
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]* learn_rate
            b -= learn_rate
        elif(y_hat - y[i] == -1):
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]* learn_rate
            b += learn_rate
    return W,b

# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 40):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines


data = np.loadtxt('/Users/bikash/work/udacity/supervised-learning/perceptron/data.csv', 
                  delimiter = ',')

X = data[:,:-1]
y = data[:,-1]

#print(X,y)

lines = trainPerceptronAlgorithm(X, y)
print('Epoch :\t\tW\t\tB')
for n, line in enumerate(lines):
    print('{}:\t\t{}\t\t{}'
          .format(str(n+1).zfill(2),
                  round(line[0][0],3), 
                  round(line[1][0],3)))