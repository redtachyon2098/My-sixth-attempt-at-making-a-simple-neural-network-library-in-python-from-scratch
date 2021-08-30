import numpy as n

import random as r

import time as t

def activate(vector, func, derivative=False):
    if func == "relu":
        dupe = n.zeros_like(vector)
        dupe[vector < 0] = 1023
        dupe += 1
        if not derivative:
            return vector / dupe
        else:
            return 1 / dupe
    if func == "sigmoid":
        if not derivative:
            return 1 / (1 + n.exp(-vector))
        else:
            return n.exp(-vector) / (1 + n.exp(-vector)) ** 2

class layer:
    def __init__(self, inDimensions, outDimensions, activation_func):
        self.invector = n.zeros(inDimensions)
        self.rawout = n.zeros(outDimensions)
        self.outvector = n.zeros(outDimensions)
        self.weights = n.random.rand(inDimensions, outDimensions)
        self.biases = n.random.rand(outDimensions)
        self.dweights = n.zeros_like(self.weights)
        self.dbiases = n.zeros_like(self.biases)
        self.dnodes = n.zeros_like(self.invector)
        self.activation = activation_func

    def run(self, vector):
        self.invector = n.array(vector)
        self.rawout = n.dot(self.invector, self.weights) + self.biases
        self.outvector = activate(self.rawout, self.activation)

    def derive(self, vector, output, fac):
        result = n.array(output)
        dout = 2 * (self.outvector - result)
        self.dweights = n.outer(self.invector, dout * activate(self.rawout, self.activation, derivative=True) * self.rawout)
        self.dbiases = dout
        self.dnodes = n.dot(dout * activate(self.rawout, self.activation, derivative=True) * self.rawout, self.weights.T)
        total = abs(n.sum(self.dweights) + n.sum(self.dbiases)) + 0.01
        self.weights -= self.dweights * fac / total
        self.biases -= self.dbiases * fac / total
        nextone = self.invector - self.dnodes / 2
        return nextone

class network:
    def __init__(self, dimensions):
        self.structure = []
        self.cost = 0
        self.CostValue = 0
        for x in range(len(dimensions) - 1):
            self.structure.append(layer(dimensions[x], dimensions[x + 1], "relu"))

    def predict(self, input_list):
        final = input_list
        for layer in self.structure:
            layer.run(final)
            final = layer.outvector
        return final

    def output(self):
        return list(self.structure[-1].outvector)

    def iterate(self, inputvector, outputvector, LearnRate):
        cost = self.predict(inputvector)
        cost -= outputvector
        cost = cost ** 2
        cost = cost.sum()
        self.cost = cost
        supposed = outputvector
        for q in range(len(self.structure)):
            x = len(self.structure) - q - 1
            supposed = self.structure[x].derive(self.structure[x].invector, supposed, LearnRate * cost)

    def train(self, inputs, outputs, LearnRate, iterations):
        clock = t.time()
        for i in range(iterations):
            c = 0
            for x, y in enumerate(inputs):
                self.iterate(y, outputs[x], LearnRate)
                c += self.cost
            self.CostValue = c / len(inputs)
            if t.time() - clock >= 2:
                print(self.CostValue)
                clock = t.time()
                
    def toomuch(self, inputss, outputss, LearnRate, iterations, minibatchlength):
        dummylist = []
        inputs = []
        outputs = []
        while len(dummylist) < min(len(inputss), minibatchlength):
            RNG = r.randint(0, len(inputss) - 1)
            if not RNG in dummylist:
                dummylist.append(RNG)
                inputs.append(inputss[RNG])
                outputs.append(outputss[RNG])
        self.train(inputs, outputs, LearnRate, iterations)
