import pandas
import numpy
import matplotlib.pyplot as plt
from pip._vendor.distlib.compat import raw_input

data = pandas.read_csv('iris.data', header=None).values  # reading from the csv file
flowerInfo = {
    'numberOfPatterns': numpy.shape(data)[0],  # dictionary array for patterns
    'numberOfAttributes': numpy.shape(data)[1]  # and attributes
}
x = data[:, :4]  # giving the values of the flowers without the name
names = data[:, 4]  # giving the names without the values
plt.plot(x[0:50, 0], x[0:50, 2],
         'b.')  # blue dot, different colors for different types that change every 50 in this case
plt.plot(x[50:100, 0], x[50:100, 2], 'r.')  # red dot
plt.plot(x[100:150, 0], x[100:150, 2], 'g.')  # green dot
plt.title('Graph of all Patterns')  # title of the graph
plt.show()  # showing the graph
t = numpy.zeros(flowerInfo['numberOfPatterns'], dtype=int)  # giving the table t the value of 150 with zeros
ans = 'y'  # giving the ans the value of y
flag = True
while ans == 'y':  # while ans is y
    print('1.Διαχωρισμός Iris-setosa από (Iris-versicolor και Iris-virginica)\n'
          '2.Διαχωρισμός Iris-virginica από (Iris-setosa και Iris-versicolor)\n'
          '3.Διαχωρισμός Iris-versicolor από (Iris-setosaκαιIris-virginica)')  # printing the menu
    userInput = int(raw_input())  # user input
    if userInput == 1:  # if user input = 1
        map_dictionary = {  # creating dictionary 1 0 0
            "Iris-setosa": 1,
            "Iris-versicolor": 0,
            "Iris-virginica": 0
        }
        for i in range(0, len(t)):  # for i from 0 to length of t
            t[i] = map_dictionary[names.item(i)]  # did stuff that I cannot explain
    elif userInput == 2:  # for i from 0 to length of t
        map_dictionary = {  # creating dictionary 0 1 0
            "Iris-setosa": 0,
            "Iris-versicolor": 1,
            "Iris-virginica": 0
        }
        for i in range(0, len(t)):
            t[i] = map_dictionary[names.item(i)]  # did stuff that I cannot explain
    elif userInput == 3:  # for i from 0 to length of t
        map_dictionary = {  # creating dictionary 0 0 1
            "Iris-setosa": 0,
            "Iris-versicolor": 0,
            "Iris-virginica": 1
        }
        for i in range(0, len(t)):
            t[i] = map_dictionary[names.item(i)]  # did stuff that I cannot explain
    if flag:
        x = numpy.hstack((x, numpy.atleast_2d(numpy.ones(150)).T))  # adding one more column of 1 in the x list
        flag = False
    xtrain = x[0:40]  # initializing the xtrain array
    xtrain = numpy.vstack((xtrain, x[50:90], x[100:140]))  # extending the array
    xtest = x[40:50]  # initializing the xtest array
    xtest = numpy.vstack((xtest, x[90:100], x[140:150]))  # extending the array
    ttrain = t[0:40]  # initializing the array
    ttrain = numpy.hstack((ttrain, t[50:90], t[100:140]))  # extending the array
    ttest = t[40:50]  # initializing the xtest array
    ttest = numpy.hstack((ttest, t[90:100], t[140:150]))  # extending the array
    plt.plot(xtrain[:, 0], xtrain[:, 2], 'b.')
    plt.plot(xtest[:, 0], xtest[:, 2], 'r.')
    plt.title('xtrain ttest patterns')
    plt.show()