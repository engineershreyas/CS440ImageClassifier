# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation

import util
PRINT = True

class MiraClassifier:
  """
  Mira classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "mira"
    self.automaticTuning = False 
    self.C = 0.001
    self.legalLabels = legalLabels
    self.max_iterations = max_iterations
    self.initializeWeightsToZero()

  def initializeWeightsToZero(self):
    "Resets the weights of each label to zero vectors" 
    self.weights = {}
    for label in self.legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
  
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    "Outside shell to call your method. Do not modify this method."  
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
    
    if (self.automaticTuning):
        Cgrid = [0.002, 0.004, 0.008]
    else:
        Cgrid = [self.C]
        
    return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
    """
    This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid, 
    then store the weights that give the best accuracy on the validationData.
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    representing a vector of values.
    """

    best_weights = {}
    best_c = -1
    best_accuracy = -1
    for c in Cgrid:
        self.initializeWeightsToZero()

        for i in range(self.max_iterations):
            for j in range(len(trainingData)):
                datum = trainingData[j]
                label = trainingLabels[j]
                predicted_label = self.classify([datum])[0]

                if predicted_label == label:
                    continue

                # tau = ((w_y' - w_y) * f + 1) / (2 * ||f||^2)
                # ||f|| = sqrt(sum((f_i,j)^2 for every feature)), ||f||^2 cancels sqrt
                tau = (((self.weights[predicted_label] - self.weights[label]) * datum) + 1.0) / (2 * sum([val * val for val in datum.values()]))
                tau = min(c, tau)

                delta = datum.copy()
                for feature in datum.keys():
                    delta[feature] *= tau

                for feature in delta.keys():
                    self.weights[label][feature] += delta[feature]
                    self.weights[predicted_label][feature] -= delta[feature]

        predictions = self.classify(validationData)
        accuracy = 0
        for i in range(len(predictions)):
            if predictions[i] == validationLabels[i]:
                accuracy += 1

        if accuracy > best_accuracy:
            best_weights = self.weights.copy()
            best_c = c
            best_accuracy = accuracy

    self.C = best_c
    self.weights = best_weights

  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns a list of the 100 features with the greatest difference in feature values
                     w_label1 - w_label2

    """
    featuresOdds = []

    for feature in self.features:
        featuresOdds.append((self.weights[label1][feature] - self.weights[label2][feature], feature))

    featuresOdds = sorted(featuresOdds, key = lambda x: x[0], reverse = True)[:100]
       
    return list(map(lambda x: x[1], featuresOdds))
