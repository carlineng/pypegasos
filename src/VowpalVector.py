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

    def innerProduct(self, vec):
        """ Input format here is similar to Vowpal Wabbit:
            <feature_name>:<feature_value> """
        innerProd = 0.
        for coord in vec:
            feature = coord.split(':')
            featureName = str(feature[0])
            featureVal = float(feature[1])

            wtVal = self.weights[self._hash(featureName)]
            radVal = self._rademacher(featureName)
            innerProd = innerProd + (featureVal * wtVal * radVal)

        return innerProd

    def _hash(self, val):
        return mmhash.get_hash(val) % self.vectorSize

    def _rademacher(self, val):
        if (mmhash.get_hash(val) >> 1) % 2 == 0:
            return 1
        else:
            return -1
