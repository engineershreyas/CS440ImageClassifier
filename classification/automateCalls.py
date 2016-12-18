import os

x = 4000
while x <= 5000:
    command = "python dataClassifier.py -c perceptron -t " + str(x)
    os.system(command)
    x = x + 500
y = 45
while y <= 405:
    command = "python dataClassifier.py -c perceptron -d faces -t " + str(y)
    os.system(command)
    y = y + 45
command = "python dataClassifier.py -c perceptron -d faces -t 451"
os.system(command)
