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

The activation is function determined by the function "activation(x)".

"b.CostValue" in this example is a number corresponding to the cost of the network. This will be explained later.

the function "predict" makes a prediction with the given network and a single-dimensional list as the input. However, it does not return the actual output.
example:

a.predict([1,1]) #makes a prediction, does not return the result.

the function "output" returns the output of the latest prediction performed by the network.
example:

a.predict([1,1]); a.output() #makes a prediction, and returns the result.

the function "cost" calculates the cost of the network, and the derivatives of the cost respectively and can be ignored. It is used when calculating the gradients.

the function "gradient" calculates the derivative of the cost function in respect to each weight and bias in the most childish way possible: by adding a small number( in this case, 10^(-8) ) and measuring the difference of the cost. This method, although it is valid, is incredibly inefficient, since it requires the time consuming "predict" function to run neumerous times( literally for every weight and bias ).

the function "train" trains the network on multiple training examples determined by the two lists "inputs", and "outputs". The first list is for the inputs, and the second list is for the corresponding desired outputs. It uses the aforementioned childish version of gradient descent. Example:

a.train([[0,0],[0,1],[1,0],[1,1]], [[0],[1],[1],[0]], 0.5, 1000) # this trains the network on the four training examples and their desired outputs.
