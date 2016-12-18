# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    self._probs = {}    # Conditional probabilities
    self._p = None      # Prior distribution
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    n = len(trainingData)

    counts = util.Counter()
    for label in trainingLabels:
        counts[label] += 1.0

    self._p = self._normalize(counts, sum(counts.values()))

    counts = {}
    for feature in self.features:
        counts[feature] = {}
        for label in self.legalLabels:
            counts[feature][label] = {
                0: 0,
                1: 0
            }

    for i in range(n):
        datum = trainingData[i]
        label = trainingLabels[i]

        for (feature, val) in datum.items():
            counts[feature][label][val] += 1.0

    best_k = -1
    best_accuracy = -1
    for k in kgrid:
        self._probs = self._calculate_cond_probs(counts, k)
        predictions = self.classify(validationData)

        accuracy = 0
        for i in range(len(predictions)):
            if predictions[i] == validationLabels[i]:
                accuracy += 1
 
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy

    self._probs = self._calculate_cond_probs(counts, best_k)
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()

    for label in self.legalLabels:
        logJoint[label] = math.log(self._p[label])

        for (feature, val) in datum.items():
            p = self._probs[feature][label][val];
            logJoint[label] += math.log(p)

    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []

    for feature in self.features:
        featuresOdds.append((self._probs[feature][label1][1] / self._probs[feature][label2][1], feature))

    featuresOdds = sorted(featuresOdds, key = lambda x: x[0], reverse = True)[:100]
       
    return list(map(lambda x: x[1], featuresOdds))

  def _calculate_cond_probs(self, counts, k):
    ret = {}
    for (feature, labels) in counts.items():
        ret[feature] = {}
        for (label, vals) in labels.items():
            ret[feature][label] = {}

            total = sum(counts[feature][label].values()) + k + k
            for (val, count) in vals.items():
                ret[feature][label][val] = (counts[feature][label][val] + k) / total

    return ret

  def _normalize(self, counts, total):
    """
    Normalizes the counts probability distribution such that
    the summation of all the probabilities sum up to one.

    param   counts: distribution to normalize
    param   total: dict or number to use to normalize

    returns:    normalized probability distribution
    """

    ret = counts

    for (k, count) in counts.items():
        ret[k] = count / total

    return ret
