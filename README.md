# My-sixth-attempt-at-making-a-simple-neural-network-library-in-python-from-scratch
This is my attempt at making a neural network from scratch. I have been working on this since 2016. The goal of this project is to create a library capable of making and training a neural network of any arbitrary size while only using the library "random".

__________________________________________________how to use_________________________________________

The class "network" makes a feed forward neural network object of any size(determined by a list of the number of nodes in each layer.)
example:

a = network([2,3,1]) #this creates a network with 2 input nodes, 1 layer of 3 hidden nodes, and one output node.

another example:

b = network([3,5,4,2]) # this creates a network with 3 input nodes, a hidden layer with 5 nodes, a hidden layer with 4 nodes, and 2 output nodes.

"b.weights" in this example is a three-dimensional list containing the weights of the network object b.

"b.biases" in this example is a two-dimensional list containing the biases of the network object b.

"b.nodes" in this example is a two dimensional list containing the values of the nodes of the network object b.

"b.raw" in this example is a two dimensional list containing the values of the nodes before it was sent through the activation function determined by the function "activation(x)". This is useful for calculating the derivative of the activation function itself, which is needed for back propagation.

"b.CostValue" in this example is a number corresponding to the cost of the network. This will be explained later.

the function "predict" makes a prediction with the given network and a single-dimensional list as the input. However, it does not return the actual output.
example:

a.predict([1,1]) #makes a prediction, does not return the result.

the function "output" returns the output of the latest prediction performed by the network.
example:

a.predict([1,1]); a.output() #makes a prediction, and returns the result.

the functions "cost" and "cost_" each calculates the cost of the network, and the derivatives of the cost respectively and can be ignored. It is used when calculating the gradients.

the function "gradient" calculates the derivative of the cost function in respect to each weight and bias by adding a small number( in this case, 10^(-8) ) and measuring the difference of the cost. This method, although it is valid, is incredibly inefficient, since it requires the time consuming "predict" function to run multiple times.

the function "backprop" calculates the derivative of the cost function in respect to each weight and bias using the chain rule( see this youtube video by 3blue1brown explaining this method in detail: https://www.youtube.com/watch?v=tIeHLnjs5U8&t=546s ). Since I am not experienced enough, this function is still produces wrong answers, therefore not training the network properly. If you know a way to fix this, let me know through email.

the function "train" trains the network using multiple training examples using gradient descent. The method used to calculate the gradient is determined by modifying the line after the first comment, changing the function name from "backprop" to "gradient", or vice versa.
example:

a.train([[0,0],[0,1],[1,0],[1,1]], [[0],[1],[1],[0]], 0.5, 1000) # this trains the network on the four training examples and their desired outputs.

the function "train_" is there for experimental purposes and can be ignored completely.
