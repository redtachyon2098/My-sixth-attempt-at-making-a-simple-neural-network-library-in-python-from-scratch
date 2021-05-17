import random as r

import time as t

alert = 1000

alertatall = True

expand = 1

expandrate = 60

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

def clone(thing):
    if(type(thing) == list):
        a = []
        for x in range(len(thing)):
            a.append(clone(thing[x]))
        return a
    else:
        return thing

def derivative(x):
    return 1e+8 * (activation(x + 1e-8) - activation(x))

class network:
    def __init__(self,nodes):
        self.costv = 0
        self.nodes = []
        self.raw = []
        self.weights = []
        self.biases = []
        a = []
        self.CostValue = 0
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
                    
    def backup(self):
        n = []
        w = []
        b = []
        for x in range(len(self.nodes)):
            n.append(len(self.nodes[x]))
        for x in range(len(self.weights)):
            for y in range(len(self.weights[x])):
                b.append(self.biases[x][y])
                for z in range(len(self.weights[x][y])):
                    w.append(self.weights[x][y][z])
        nfile = open("n.txt", "w")
        bfile = open("b.txt", "w")
        wfile = open("w.txt", "w")
        a = ""
        for x in range(len(n)):
            a += str(n[x]) + ","
        nfile.write(a)
        a = ""
        for x in range(len(w)):
            a += str(w[x]) + ","
        wfile.write(a)
        a = ""
        for x in range(len(b)):
            a += str(b[x]) + ","
        bfile.write(a)
        nfile.close()
        bfile.close()
        wfile.close()

    def load(self):
        nfile = open("n.txt", "r")
        bfile = open("b.txt", "r")
        wfile = open("w.txt", "r")
        n = nfile.read()
        b = bfile.read()
        w = wfile.read()
        dimensions = []
        weightnumbers = []
        biasnumbers = []
        nodes = []
        weights = []
        biases = []
        a = ""
        for x in range(len(n)):
            if(n[x] != ","):
                a += n[x]
            else:
                dimensions.append(int(a))
                a = ""
        a = ""
        for x in range(len(b)):
            if(b[x] != ","):
                a += b[x]
            else:
                biasnumbers.append(float(a))
                a = ""
        a = ""
        for x in range(len(w)):
            if(w[x] != ","):
                a += w[x]
            else:
                weightnumbers.append(float(a))
                a = ""
        for x in range(len(dimensions)):
            nodes.append([])
            for y in range(dimensions[x]):
                nodes[x].append(0)
        p = 0
        q = 0
        for x in range(len(dimensions) - 1):
            weights.append([])
            biases.append([])
            for y in range(dimensions[x + 1]):
                biases[x].append(biasnumbers[p])
                p += 1
                weights[x].append([])
                for z in range(dimensions[x]):
                    weights[x][y].append(weightnumbers[q])
                    q += 1
        self.nodes = nodes
        self.raw = nodes
        self.weights = weights
        self.biases = biases

    def predict(self,input_list):
        self.nodes[0] = input_list
        self.raw[0] = input_list
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
        self.costv = b
        return b

    def backprop(self, input_list, output_list):
        costnumber = self.cost(input_list, output_list)
        self.predict(input_list)
        w = clone(self.weights)
        b = clone(self.biases)
        expectedoutput = output_list
        for p in range(len(self.nodes) - 1):
            x = len(self.nodes) - p - 1
            differences = []
            for y in range(len(self.nodes[x])):
                differences.append(self.nodes[x][y] - expectedoutput[y])
            for y in range(len(self.nodes[x])):
                b[x - 1][y] = 2 * differences[y] * derivative(self.raw[x][y])
                for z in range(len(self.nodes[x - 1])):
                    w[x - 1][y][z] = self.nodes[x - 1][z] * 2 * differences[y] * derivative(self.raw[x][y])
            expectedoutput = []
            for y in range(len(self.nodes[x - 1])):
                a = 0
                for z in range(len(self.nodes[x])):
                    a += self.weights[x - 1][z][y] * 2 * differences[z] * derivative(self.raw[x][z])
                expectedoutput.append(((a / len(self.nodes[x])) / (-2)) + self.nodes[x - 1][y])
        return [w,b]

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
                c = self.backprop(inputs[r],outputs[r])
                avgCost += self.costv / len(inputs)
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
            self.CostValue = lastC
            if(int(q / alert) == q / alert and alertatall == True):
                print("Iteration:", q, "loss:",avgCost, "Time:", int((t.time() - clock) / 0.06) / 1000, "minutes, improvement:", improvement)
        if(alertatall == True):
            print(improvement / l)
            print("final loss:",avgCost)
            
    def toomuch(self,inputss,outputss,LearnRate,iterations, minibatchlength):
        clock = t.time()
        l = LearnRate
        lastC = 0
        for q in range(iterations):
            dummylist = []
            inputs = []
            outputs = []
            while len(inputs) < min(len(inputss), minibatchlength):
                RNG = r.randint(0, len(inputss) - 1)
                if not RNG in dummylist:
                    dummylist.append(RNG)
                    inputs.append(inputss[RNG])
                    outputs.append(outputss[RNG])
            avgw = []
            avgb = []
            total = 0
            avgCost = 0
            for x in range(len(self.weights)):
                avgw.append([])
                avgb.append([])
                for y in range(len(self.weights[x])):
                    avgw[x].append([])
                    avgb[x].append(0)
                    for z in range(len(self.weights[x][y])):
                        avgw[x][y].append(0)
            for j in range(len(inputs)):
                c = self.backprop(inputs[j],outputs[j])
                avgCost += self.costv / len(inputs)
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
                    self.biases[x][y] -= avgb[x][y] * l * (avgCost ** 0.5) / total
                    #self.biases[x][y] -= avgb[x][y] * l / total
                    for z in range(len(self.weights[x][y])):
                        self.weights[x][y][z] -= avgw[x][y][z] * l * (avgCost ** 0.5) / total
                        #self.weights[x][y][z] -= avgw[x][y][z] * l / total
            improvement = lastC - avgCost
            lastC = avgCost
            self.CostValue = lastC
            if(int(q / alert) == q / alert and alertatall == True):
                print("Iteration:", q, "loss:",avgCost, "Time:", int((t.time() - clock) / 0.06) / 1000, "minutes, improvement:", improvement)
        if(alertatall == True):
            print("final loss:",avgCost)
            print(improvement / l)
            
    def expandlayer(self):
        a = []
        b = []
        c = []
        for x in range(len(self.nodes[len(self.nodes) - 1])):
            a.append(0)
            b.append([])
            c.append(0)
            for y in range(len(self.nodes[len(self.nodes) - 1])):
                if(x == y):
                    b[x].append(1)
                else:
                    b[x].append(0)
        self.nodes.append(a)
        self.weights.append(b)
        self.biases.append(c)

    def expandnode(self):
        if(len(self.nodes) == 2):
            self.expandlayer()
        else:
            theluck = r.randint(1, len(self.nodes) - 2)
            self.nodes[theluck].append(0)
            self.biases[theluck - 1].append(0)
            for x in range(len(self.weights[theluck])):
                self.weights[theluck][x].append(r.random())
            a = []
            for x in range(len(self.nodes[theluck - 1])):
                a.append(0)
            self.weights[theluck - 1].append(a)

    def deviate(self, count, randomness):
        w = []
        b = []
        for p in range(count - 1):
            w.append([])
            b.append([])
            for x in range(len(self.weights)):
                w[p].append([])
                b[p].append([])
                for y in range(len(self.weights[x])):
                    w[p][x].append([])
                    b[p][x].append(self.biases[x][y] + randomness * ((2 * (r.random() - 0.5)) ** 3))
                    for z in range(len(self.weights[x][y])):
                        w[p][x][y].append(self.weights[x][y][z] + randomness * ((2 * (r.random() - 0.5)) ** 3))
        w.append(self.weights)
        b.append(self.biases)
        return w, b
    
    def refine(self, inputs, outputs, unitcount, randomness, iterations):
        starttime = t.time()
        rand = randomness
        prevcost = 0
        count = 0
        for p in range(iterations):
            w = []
            b = []
            w, b = self.deviate(unitcount, rand)
            costs = []
            for q in range(len(w)):
                self.weights = w[q]
                self.biases = b[q]
                avgcost = 0
                for x in range(len(inputs)):
                    avgcost += self.cost(inputs[x], outputs[x])
                avgcost /= len(inputs)
                costs.append(avgcost)
            selection = smallest(costs)
            self.weights = w[selection]
            self.biases = b[selection]
            if(p % alert == 0 and alertatall == True):
                print("Iteration: ", p, " Cost: ", costs[selection], "Time: ", int(((t.time() - starttime) / 60) * 1000) / 1000, " minutes  Randomness: ", rand)
            if(p != 0 and prevcost != costs[selection]):
                count = 0
                #rand /= 1.05
            if(count % 30 == 29):
                rand /= 1.05
            if(expand == 1):
                if(count == expandrate):
                    rand = randomness
                    if(alertatall == True):
                        print("\n   expanded!\n")
                    lel = r.randint(0,20)
                    if(lel == 0):
                        self.expandlayer()
                    else:
                        self.expandnode()
            prevcost = costs[selection]
            count += 1
