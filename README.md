# My-sixth-attempt-at-making-a-simple-neural-network-library-in-python-from-scratch
This is my attempt at making a neural network from scratch. I have been working on this since 2016. The goal of this project is to create a library capable of making and training a neural network of any arbitrary size while only using the library "random".

__________________________________________________functions__________________________________________________

The class "network" makes a feed forward neural network object of any size(determined by a list of the number of nodes in each layer.)
example:

a = network([2,3,1]) #this creates a network with 2 input nodes, 1 layer of 3 hidden nodes, and one output node.

another example:

b = network([3,5,4,2]) # this creates a network with 3 input nodes, a hidden layer with 5 nodes, a hidden layer with 4 nodes, and 2 output nodes.

"b.weights" in this example is a three-dimensional list containing the weights of the network object b.

"b.biases" in this example is a two-dimensional list containing the biases of the network object b.

"b.nodes" in this example is a two dimensional list containing the values of the nodes of the network object b.

"b.raw" in this example is a two dimensional list containing the non-activated values of the nodes of the network object b.

The activation is function determined by the function "activation(x)".

"b.CostValue" in this example is a number corresponding to the cost of the network. This will be explained later.

the function "predict" makes a prediction with the given network and a single-dimensional list as the input. However, it does not return the actual output.
example:

a.predict([1,1]) #makes a prediction, does not return the result.

the function "output" returns the output of the latest prediction performed by the network.
example:

a.predict([1,1]); a.output() #makes a prediction, and returns the result.

the function "cost" calculates the cost of the network, and the derivatives of the cost respectively and can be ignored. It is used when calculating the gradients.

the function "backprop" calculates the derivative of the cost function in respect to each weight and bias using back propogation.

the function "train" trains the network on multiple training examples determined by the two lists "inputs", and "outputs". The first list is for the inputs, and the second list is for the corresponding desired outputs. It uses the aforementioned primitive version of gradient descent. Example:

a.train([[0,0],[0,1],[1,0],[1,1]], [[0],[1],[1],[0]], 0.5, 1000) # this trains the network on the training examples and their desired outputs. In this case, the classic XOR truth table.

You can toggle whether the code prints various information to the console by changing the variable "alertatall" to be True or False.

"toomuch" is the same as "train" but it only trains the network on a few randomly chosen training examples at a time to speed up the process. Because of this function, there isn't any real need to use "train" anymore, but it is still there for backwards compatibility.

There are additional functions. "backup()" creates three text files in the same location as the code itself which together, carries the information necessary for perfectly reconstructing the neural network(The weights, and the biases. The values in the nodes are not stored however, because it is not needed for the network to function.) This is used in order to be able to train the network, stop it, and then keep training it without losing any progress, even if you reboot your computer. Note: The files "backup()" creates have the names "n", "w", "b" by default. Make sure you don't have another text file with those names, or it may become overwritten.

a.backup() #Backs up the network into three text files.

"load()" reads from the three text files created by backup() and creates a neural network which has the same weights and biases as the previously backed up network. The values of the nodes are not persistens, and they will be zero'ed out.

a.load() #Restores the previously backed up network.

"expandlayer()" adds another layer on top of the neural network without changing its inputs and outputs.

a.expandlayer() #Expands the network by one layer.

"expandnode()" adds a node to a random layer without changing its inputs and outputs.

a.expandnode() #Expands the network by one layer.

"deviate(count, randomness)" creates several randomly-altered versions of the network without changing the original. Exactly how many versions are created is determined by the first argument.("count" in this case.) The maximum value a weight or bias can be changed by is determined by the second argument.("randomness" in this case.)

a.deviate(10, 1) #returns 10 slightly altered versions of the original network, each weight and bias being changed by values ranging from -1 to 1.

"refine(input, outputs, unitcount, randomness, iterations)" refines the network by creating a number of slightly altered networks and choosing the best one, then altering that new one, over and over. For big networks, this is faster than "train".

a.refine([[0,0],[0,1],[1,0],[1,1]], [[0],[1],[1],[0]], 100, 0.5, 1000) #This creates 100 different versions of the original network(each differing from the original by 0.5) and chooses the best one, 1000 times.
