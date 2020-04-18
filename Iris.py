# imports
import pandas
import numpy
import matplotlib.pyplot as plt
from pip._vendor.distlib.compat import raw_input


def perceptron(x, t, maxepochs, beta):  # perceptron function
    #  w = numpy.random.uniform(low=-1, high=1, size=len(x[0]))
    w = numpy.random.randn(len(x[0]), 1)  # filling the w by random numbers
    flag = True  # initializing the flag
    epoch = 1  # initializing the epoch
    u = numpy.zeros(len(x))  # initializing the u array
    predict = numpy.zeros(len(x))  # initializing the predict list
    while maxepochs >= epoch & flag:
        flag = False
        for i in range(len(x)):  # for each i in x len
            u[i] = sum(w[j] * x[i, j] for j in range(len(x[0])))  # finding the stimulation
            if u[i] < 0:  # if u[0] is lower than 0
                predict[i] = 0  # we give predict the value of 0
            else:  # in any other case
                predict[i] = 1  # we give predict the value of 1
            if predict[i] != t[i]:  # if predict is not equal to t
                for j in range(0, len(w)):  # for each w
                    w[j] = w[j] + beta * (t[i] - predict[i]) * x[i, j]  # giving new value to our w
                flag = True  # if we find an error, we make the flag true again
        epoch += 1
    print(' i am here')
    plt.figure()  # creating new figure
    plt.plot(predict, 'ro', t, 'b.')  # red circles for the predict array and blue dots for the t array
    plt.title('perceptron view')  # just the title of our plt figure
    plt.show()  # showing our plt figure
    return w  # returning the w


if __name__ == '__main__':
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
        plt.plot(xtrain[:, 0], xtrain[:, 2], 'b.')  # creating the diagrams
        plt.plot(xtest[:, 0], xtest[:, 2], 'r.')  # creating the diagrams
        plt.title('xtrain ttest patterns')  # giving a title to our graph
        plt.show()  # showing the graph
        while True:
            print('1. Υλοποίηση με Perceptron\n' +
                  '2. Υλοποίηση με Adaline\n' +
                  '3. Υλοποίηση με Λύση Ελαχίστων Τετραγώνων\n' +
                  '4. Επιστροφή στο αρχικό Menu')  # printing the available options
            userInput = int(raw_input())  # user input
            if userInput == 4:  # if userInput is 4, we break the while
                break
            elif userInput == 1:
                maxepochs = int(raw_input('Max Epochs:'))  #
                beta = float(raw_input('Beta:'))
                w = perceptron(xtrain, ttrain, maxepochs, beta)
