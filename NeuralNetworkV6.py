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

    def train(self,inputs,outputs,LearnRate,iterations):
        for q in range(iterations):
            avgw = []
            avgb = []
            total = 0
            avgCost = 0
            backW = self.weights
            backB = self.biases
            for x in range(len(self.weights)):
                avgw.append([])
                avgb.append([])
                for y in range(len(self.weights[x])):
                    avgw[x].append([])
                    avgb[x].append(0)
                    for z in range(len(self.weights[x][y])):
                        avgw[x][y].append(0)
            for r in range(len(inputs)):
                c = self.gradient(inputs[r],outputs[r])
                avgCost += self.CostValue / len(inputs)
                for x in range(len(self.weights)):
                    for y in range(len(self.weights[x])):
                        avgb[x][y] += c[1][x][y] / len(inputs)
                        total += c[1][x][y]
                        for z in range(len(self.weights[x][y])):
                            avgw[x][y][z] += c[0][x][y][z] / len(inputs)
                            total += c[0][x][y][z]
            if(total < 0):
                total = -total
            if(total == 0):
                total = 1e-256
            for x in range(len(self.weights)):
                for y in range(len(self.weights[x])):
                    self.biases[x][y] -= avgb[x][y] * LearnRate * (avgCost)**0.5 / total
                    for z in range(len(self.weights[x][y])):
                        self.weights[x][y][z] -= avgw[x][y][z] * LearnRate * (avgCost)**0.5 / total
            if(q > 1 and avgCost > lastC):
                self.weights = backW
                self.biases = backB
                print(q,lastC,avgCost)
                break
            lastC = avgCost
            if(int(q / 50) == q / 50):
                print("Iteration:", q, "loss:",avgCost)
        print("final loss:",avgCost)
