import math
from copy import deepcopy


class MaxEntropy:
    def __init__(self, EPS=0.005):
        self._samples = []
        self._Y = set()  # label set, equivalent to deduplicated y
        self._numXY = {}  # key is (x,y), value is occurrence count
        self._N = 0  # number of samples
        self._Ep_ = []   # expected value of features under empirical distribution
        self._xyID = {}   # key stores (x,y), value stores id number
        self._n = 0  # number of feature key-value pairs (x,y)
        self._C = 0   # maximum number of features
        self._IDxy = {}    # key is id number, value is corresponding (x,y)
        self._w = []
        self._EPS = EPS   # convergence condition
        self._lastw = []    # previous w parameter values

    def loadData(self, dataset):
        self._samples = deepcopy(dataset)
        for items in self._samples:
                y = items[0]
                X = items[1:]
                self._Y.add(y)  # if y already exists in the set, it is automatically ignored
                for x in X:
                    if (x, y) in self._numXY:
                        self._numXY[(x, y)] += 1
                    else:
                        self._numXY[(x, y)] = 1

        self._N = len(self._samples)
        self._n = len(self._numXY)
        self._C = max([len(sample)-1 for sample in self._samples])
        self._w = [0]*self._n
        self._lastw = self._w[:]

        self._Ep_ = [0] * self._n
        for i, xy in enumerate(self._numXY):   # compute the expectation of feature function fi under empirical distribution
            self._Ep_[i] = self._numXY[xy]/self._N
            self._xyID[xy] = i
            self._IDxy[i] = xy

    def _Zx(self, X):    # compute each Z(x) value
        zx = 0
        for y in self._Y:
            ss = 0
            for x in X:
                if (x, y) in self._numXY:
                    ss += self._w[self._xyID[(x, y)]]
            zx += math.exp(ss)
        return zx

    def _model_pyx(self, y, X):   # compute each P(y|x)
        zx = self._Zx(X)
        ss = 0
        for x in X:
            if (x, y) in self._numXY:
                ss += self._w[self._xyID[(x, y)]]
        pyx = math.exp(ss)/zx
        return pyx

    def _model_ep(self, index):   # compute the expectation of feature function fi under the model
        x, y = self._IDxy[index]
        ep = 0
        for sample in self._samples:
            if x not in sample:
                continue
            pyx = self._model_pyx(y, sample)
            ep += pyx/self._N
        return ep

    def _convergence(self):  # check whether all parameters have converged
        for last, now in zip(self._lastw, self._w):
            if abs(last - now) >= self._EPS:
                return False
        return True

    def predict(self, X):   # compute prediction probabilities
        Z = self._Zx(X)
        result = {}
        for y in self._Y:
            ss = 0
            for x in X:
                if (x, y) in self._numXY:
                    ss += self._w[self._xyID[(x, y)]]
            pyx = math.exp(ss)/Z
            result[y] = pyx
        return result

    def train(self, maxiter=1000):   # train the model
        for loop in range(maxiter):  # maximum number of training iterations
            print("iter:%d" % loop)
            self._lastw = self._w[:]
            for i in range(self._n):
                ep = self._model_ep(i)    # compute the model expectation for the i-th feature
                self._w[i] += math.log(self._Ep_[i]/ep)/self._C   # update parameters
            print("w:", self._w)
            if self._convergence():  # check for convergence
                break


dataset = [['no', 'sunny', 'hot', 'high', 'FALSE'],
           ['no', 'sunny', 'hot', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'high', 'FALSE'],
           ['yes', 'rainy', 'mild', 'high', 'FALSE'],
           ['yes', 'rainy', 'cool', 'normal', 'FALSE'],
           ['no', 'rainy', 'cool', 'normal', 'TRUE'],
           ['yes', 'overcast', 'cool', 'normal', 'TRUE'],
           ['no', 'sunny', 'mild', 'high', 'FALSE'],
           ['yes', 'sunny', 'cool', 'normal', 'FALSE'],
           ['yes', 'rainy', 'mild', 'normal', 'FALSE'],
           ['yes', 'sunny', 'mild', 'normal', 'TRUE'],
           ['yes', 'overcast', 'mild', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'normal', 'FALSE'],
           ['no', 'rainy', 'mild', 'high', 'TRUE']]

maxent = MaxEntropy()
x = ['overcast', 'mild', 'high', 'FALSE']
maxent.loadData(dataset)
maxent.train()
print('predict:', maxent.predict(x))
