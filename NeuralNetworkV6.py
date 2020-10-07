import random as r

def sigmoid(x):
    try:
        return 1 / (1 + (2.72828 ** (-x)))
    except OverflowError:
        return 0

def relu(x):
    if(x > 0):
        return x
    else:
        return 0

def leakrelu(x):
    if(x > 0):
        return x
    else:
        return 0.05 * x

def inverse(x):
    if(x == 0):
        x = 1e-07
    return 1 / x

def step(x):
    if(x < 0):
        return 0
    else:
        return 1

def activation(x):
    return leakrelu(x)

def average(numbers):
    b = 0
    for x in range(len(numbers)):
        b += numbers[x]
    return b / len(numbers)

class network:
    def __init__(self,nodes):
        self.CostValue = 0
        self.nodes = []
        self.weights = []
        self.biases = []
        self.raw = []
        a = []
        for x in range(nodes[0]):
            a.append(0)
        self.nodes.append(a)
        self.raw.append(a)
        for x in range(len(nodes) - 1):
            self.nodes.append([])
            self.raw.append([])
            self.biases.append([])
            self.weights.append([])
            for y in range(nodes[x + 1]):
                self.nodes[x + 1].append(0)
                self.raw[x + 1].append(0)
                self.biases[x].append(r.random())
                self.weights[x].append([])
                for z in range(nodes[x]):
                    self.weights[x][y].append(r.random())

    def predict(self,input_list):
        self.nodes[0] = input_list
        for x in range(len(self.biases)):
            a = []
            c = []
            for y in range(len(self.biases[x])):
                b = self.biases[x][y]
                for z in range(len(self.weights[x][y])):
                    b += self.weights[x][y][z] * self.nodes[x][z]
                a.append(activation(b))
                c.append(b)
            self.nodes[x + 1] = a
            self.raw[x + 1] = c

    def output(self):
        return self.nodes[len(self.nodes) - 1]

    def cost(self,input_list,output_list):
        self.predict(input_list)
        a = self.output()
        b = 0
        for x in range(len(a)):
            try:
                b += ((a[x] - output_list[x]) ** 2)
            except OverflowError:
                b += 16e+256
        self.CostValue = b
        return b

    def cost_(self,input_list,output_list):
        self.predict(input_list)
        a = self.output()
        b = []
        for x in range(len(a)):
            try:
                b.append(2 * (a[x] - output_list[x]))
            except OverflowError:
                b.append(16e+256)
        self.CostValue = average(b)
        return b

    def gradient(self,input_list,output_list):
        a = self.cost(input_list,output_list)
        b = [[],[]]
        for x in range(len(self.biases)):
            b[0].append([])
            b[1].append([])
            for y in range(len(self.biases[x])):
                b[0][x].append([])
                self.biases[x][y] += 1e-8
                c = self.cost(input_list,output_list)
                self.biases[x][y] -= 1e-8
                b[1][x].append(1e+8 * (c - a))
                for z in range(len(self.weights[x][y])):
                    self.weights[x][y][z] += 1e-8
                    c = self.cost(input_list,output_list)
                    self.weights[x][y][z] -= 1e-8
                    b[0][x][y].append(1e+8 * (c - a))
        return b

    def backprop(self,input_list,output_list):
        a = []
        b = []
        deriv = []
        for x in range(len(self.nodes)):
            deriv.append([])
            for y in range(len(self.nodes[x])):
                deriv[x].append(0)
        for t in range(len(self.nodes) - 1):
            x = t + 1
            for y in range(len(self.nodes[x])):
                deriv[x][y] = (activation(self.raw[x][y] + 0.00001) - self.nodes[x][y]) * 100000
        for x in range(len(self.weights)):
            a.append([])
            b.append([])
            for y in range(len(self.weights[x])):
                a[x].append([])
                b[x].append(0)
                for z in range(len(self.weights[x][y])):
                    a[x][y].append(0)
        costs = self.cost_(input_list,output_list)
        for q in range(len(deriv[len(deriv) - 1])):
            deriv[len(deriv) - 1][q] *= costs[q]
        for q in range(len(self.weights)):
            x = len(self.weights) - q - 1
            for y in range(len(self.weights[x])):
                b[x][y] = deriv[x + 1][y]
                for z in range(len(self.weights[x][y])):
                    a[x][y][z] = self.nodes[x][z] * deriv[x + 1][y]
            for y in range(len(deriv[x])):
                buffer = 0
                for z in range(len(deriv[x + 1])):
                    buffer += self.weights[x][z][y] * deriv[x + 1][z]
                deriv[x][y] *= buffer
        return [a,b]

    def train(self,inputs,outputs,LearnRate,iterations):
        for q in range(iterations):
            for r in range(len(inputs)):
                #Line used to determine calculation method
                c = self.gradient(inputs[r],outputs[r])
                for x in range(len(self.weights)):
                    for y in range(len(self.weights[x])):
                        self.biases[x][y] -= c[1][x][y] * LearnRate / len(inputs)
                        for z in range(len(self.weights[x][y])):
                            self.weights[x][y][z] -= c[0][x][y][z] * LearnRate / len(inputs)
            #if(int(q / 10) == q / 10):
            #    print(self.backprop(inputs[r],outputs[r]))
            #    print(" ")
            #    print(self.gradient(inputs[r],outputs[r]))
            #    print(" ")
            #    print(" ")
            #    print(" ")
            #    print(" ")
            if(int(q / 10) == q / 10):
                print("Iteration:", q, "loss:",self.CostValue)
        print("final loss:",self.CostValue)

    def train_(self,inputs,outputs,LearnRate,iterations):
        for q in range(iterations):
            a = self.weights
            b = self.biases
            totalgrad = 0
            AverageCost = 0
            for m in range(len(inputs)):
                AverageCost += self.cost(inputs[m],outputs[m]) / len(inputs)
            for x in range(len(self.weights)):
                for y in range(len(self.weights[x])):
                    for z in range(len(self.weights[x][y])):
                        self.weights[x][y][z] += 0.0001
                        Value = 0
                        for m in range(len(inputs)):
                            Value += self.cost(inputs[m],outputs[m])
                        Value /= len(inputs)
                        self.weights[x][y][z] -= 0.0001
                        a[x][y][z] = (Value - AverageCost) * 10000
                        totalgrad += a[x][y][z]
                    self.biases[x][y] += 0.0001
                    Value = 0
                    for m in range(len(inputs)):
                        Value += self.cost(inputs[m],outputs[m])
                    Value /= len(inputs)
                    self.biases[x][y] -= 0.0001
                    b[x][y] = (Value - AverageCost) * 10000
                    totalgrad += b[x][y]
            for x in range(len(self.biases)):
                for y in range(len(self.biases[x])):
                    for z in range(len(self.weights[x][y])):
                        self.weights[x][y][z] -= LearnRate * AverageCost * a[x][y][z] / totalgrad
                    self.biases[x][y] -= LearnRate * AverageCost * b[x][y] / totalgrad
            #if(int(q / 1) == q / 1):
            #    print("loss:",AverageCost)
        #print("final loss:",AverageCost)

#_________________________________________________________________________

r.seed(10)
a = network([2,3,1])
#a.train([[0,0],[0,1],[1,0],[1,1]],[[0],[1],[1],[0]],1,10000)
