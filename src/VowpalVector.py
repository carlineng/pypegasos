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
        maxNorm = 1./(self.lamb ** .5)
        scalar = maxNorm/self.vectorSize
        weights = [(random.random() - .5) * scalar for x in range(0, self.vectorSize)]
        return weights

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

    def updateWeights(self, lamb, t, k, updateSet):
        stepKeys = self.gradientStep(lamb, t, k, updateSet)
        self.projectStep(lamb, stepKeys)

    def gradientStep(self, lamb, t, k, updateSet):
        eta = 1./(lamb * t)
        stepDirection = {}
        for obs in updateSet:
            if len(obs) < 2: continue
            
            y = int(obs[0])

            features = obs[1:]
            for coord in features:
                feature = coord.split(':')
                featureName = str(feature[0])
                featureVal = float(feature[1]) * y
                if featureName not in stepDirection:
                    stepDirection[featureName] = 0.

                stepDirection[featureName] += featureVal

        scaling = eta/k

        for key,val in stepDirection.iteritems():
            indx = self._hash(key)
            rad = self._rademacher(key)

            self.weights[indx] *= (1 - (eta * lamb))
            self.weights[indx] += scaling * val * rad
        
        return stepDirection.keys()

    def projectStep(self, lamb, stepKeys):

        normSquared = 0.
        for feature in stepKeys:
            indx = self._hash(feature)
            normSquared += self.weights[indx] ** 2

        scaling = min(1, (1/(lamb ** .5))/(normSquared ** .5))
        if scaling != 1:
            for feature in stepKeys:
                indx = self._hash(feature)
                self.weights[indx] *= scaling
            

    def _hash(self, val):
        return mmhash.get_hash(val) % self.vectorSize

    def _rademacher(self, val):
        if (mmhash.get_hash(val) >> 1) % 2 == 0:
            return 1
        else:
            return -1

    def _norm(self):
        sqNorm = sum(map(lambda x: x*x, self.weights))
        return sqNorm ** .5
