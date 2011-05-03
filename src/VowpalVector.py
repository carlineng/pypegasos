#!/usr/bin/env python

import mmhash
import random

class VowpalVector:

    def __init__(self, nbits=18, lamb=.1):
        self.nbits = nbits
        self.lamb = lamb

        self.vectorSize = 2 ** nbits
        self.weights = self.resetWeights()

    def resetWeights(self):
        maxNorm = 1./(lamb ** .5)
        scalar = maxNorm/self.vectorSize
        self.weights = [(random.random() - .5) * scalar for x in range(0, vectorSize)]

