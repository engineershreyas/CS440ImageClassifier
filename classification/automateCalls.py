import os

def callBasedOnClassifier(classifier):
    if classifier == "mira" or classifier == "naiveBayes":
        classifier = classifier + " -a"
    if classifier == "mira":
        i = 1
        while i <= 3:
            classifier = classifier + " -i " + str(i)
            doCalls(classifier)
            i = i + 2
    else:
        doCalls(classifier)

def doCalls(classifier):
    x = 500
    while x <= 5000:
        command = "python dataClassifier.py -c " + classifier + " -f -t " + str(x)
        os.system(command)
        x = x + 500
    y = 45
    while y <= 405:
        command = "python dataClassifier.py -c " + classifier + " -f -d faces -t " + str(y)
        os.system(command)
        y = y + 45
    command = "python dataClassifier.py -c" + classifier + " -f -d faces -t 451"
    os.system(command)

#classifiers = ["naiveBayes","perceptron","mira"]
classifiers = ["mira"]
#overrite existing log.txt
f = open('log.txt','w')
f.close()
for c in classifiers:
    f = open('log.txt','a')
    f.write(c + " ")
    f.close()
    callBasedOnClassifier(c)
