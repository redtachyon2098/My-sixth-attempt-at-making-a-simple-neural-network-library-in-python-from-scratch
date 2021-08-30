# My-sixth-attempt-at-making-a-simple-neural-network-library-in-python-from-scratch: Numpy Edition
This branch is a compatible version made with the "numpy" library. It works on completely different principles, and it is missing a few features(i.e. The load/save feature), but it is largely compatible with the main version.
__________________________________________________functions__________________________________________________

The class "network" makes a feed forward neural network object of any size(determined by a list of the number of nodes in each layer.)
example:

a = network([2,3,1]) #this creates a network with 2 input nodes, 1 layer of 3 hidden nodes, and one output node.

another example:

b = network([3,5,4,2]) # this creates a network with 3 input nodes, a hidden layer with 5 nodes, a hidden layer with 4 nodes, and 2 output nodes.

the network object has a list called "structure", which contains "layer" objects. There "layer" objects are what's actually doing the work.

a.predict([1,1]) #makes a prediction, returns the result. The old version didn't use to, but now it does.

the function "output" returns the output of the latest prediction performed by the network. This was kept in for compatibility.

the function "train" trains the network on multiple training examples determined by the two lists "inputs", and "outputs". The first list is for the inputs, and the second list is for the corresponding desired outputs. It uses a primitive version of gradient descent. Example:

a.train([[0,0],[0,1],[1,0],[1,1]], [[0],[1],[1],[0]], 0.5, 1000) # this trains the network on the training examples and their desired outputs. In this case, the classic XOR truth table.

If you want to train on a single example, use the function "iterate".

"toomuch" is the same as "train" but it only trains the network on a few randomly chosen training examples at a time to speed up the process. Because of this function, there isn't any real need to use "train" anymore, but it is still there for backwards compatibility.

The load/save features isn't implemented yet, and the alternative training methods are obsolete, and are therefore cut. It's still there in the original version, however.
