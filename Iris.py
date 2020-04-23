# imports
import pandas
import numpy
import matplotlib.pyplot as plt
from pip._vendor.distlib.compat import raw_input
import sklearn.model_selection


def perceptron(x, t, maxepochs, beta):  # perceptron function
    w = numpy.random.randn(len(x[0]), 1)  # filling the w by random numbers
    flag = True  # initializing the flag
    epoch = 1  # initializing the epoch
    u = numpy.zeros(len(x))  # initializing the u array
    predict = numpy.zeros(len(x))  # initializing the predict list
    while (maxepochs >= epoch) & flag:
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
    plt.figure()  # creating new figure
    plt.plot(predict, 'ro', t, 'b.')  # red circles for the predict array and blue dots for the t array
    plt.title('perceptron view')  # just the title of our plt figure
    plt.show()  # showing our plt figure


def adaline(x, t, maxepochs, beta, minmse):
    w = numpy.random.randn(len(x[0]), 1)  # filling the w by random numbers
    epoch = 1  # initializing the epoch
    u = numpy.zeros(len(x))  # initializing the u array
    predict = numpy.zeros(len(x))  # initializing the predict list
    while maxepochs >= epoch:
        error = 0
        for i in range(len(x)):  # for each i in x len
            u[i] = sum(w[j] * x[i, j] for j in range(len(x[0])))  # finding the stimulation
            if u[i] < 0:  # if u[0] is lower than 0
                predict[i] = -1  # we give predict the value of 0
            else:  # in any other case
                predict[i] = 1  # we give predict the value of 1
            error = error + (t[i] - predict[i]) ** 2  # adding the error for each loop
            for j in range(0, len(w)):  # for each w
                w[j] = w[j] + beta * (t[i] - predict[i]) * x[i, j]  # giving new value to our w
        if error / len(x) <= minmse:
            break
        epoch += 1
    plt.figure()  # creating new figure
    plt.plot(predict, 'ro', t, 'b.')  # red circles for the predict array and blue dots for the t array
    plt.title('adaline view')  # just the title of our plt figure
    plt.show()  # showing our plt figure


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
                "Iris-versicolor": 0,
                "Iris-virginica": 1
            }
            for i in range(0, len(t)):
                t[i] = map_dictionary[names.item(i)]  # did stuff that I cannot explain
        elif userInput == 3:  # for i from 0 to length of t
            map_dictionary = {  # creating dictionary 0 0 1
                "Iris-setosa": 0,
                "Iris-versicolor": 1,
                "Iris-virginica": 0
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
                maxepochs = int(raw_input('Max Epochs:'))  # max seasons
                beta = float(raw_input('Beta:'))  # beta variable
                perceptron(xtrain, ttrain, maxepochs, beta)  # calling the perceptron for both xtrain, ttrain
                perceptron(xtest, ttest, maxepochs, beta)  # and xtest, ttest
                for i in range(9):
                    x_train, x_test, t_train, t_test = sklearn.model_selection.train_test_split(x, t,
                                                                                                test_size=0.1)  # train_test_split
                    plt.figure()  # creating a new view or something
                    plt.subplot(211)  # just like a plot, but different, rows, cols
                    plt.plot(x_train[:, 0], x_train[:, 2], "b.")  # x, y, color
                    plt.title("x_train with Train_Test_Split Perceptron")  # the title of our plt graph
                    plt.subplot(211)  # just like a plot, but different, rows, cols
                    plt.plot(x_test[:, 0], x_test[:, 2], "r.")  # x, y, color
                    plt.title("x_test with Train_Test_Split Perceptron")  # the title of our plt graph
                    plt.tight_layout()  # to make it appear normal
                    plt.show()  # showing the graph
                    perceptron(x_train, t_train, maxepochs, beta)  # calling the perceptron for both x_train, t_train
                    perceptron(x_test, t_test, maxepochs, beta)  # calling the perceptron for both x_test, t_test

            elif userInput == 2:
                ttrain2 = [i if i != 0 else -1 for i in ttrain]  # changing 0 values to -1
                ttest2 = [i if i != 0 else -1 for i in ttest]  # changing 0 values to -1
                maxepochs = int(raw_input('Max Epochs:'))  # max seasons
                beta = float(raw_input('Beta:'))  # beta variable
                minmse = float(raw_input('MinMSE:'))  # minmse input
                adaline(xtrain, ttrain2, maxepochs, beta, minmse)  # calling the adaline for both xtrain and ttrain
                adaline(xtest, ttest2, maxepochs, beta, minmse)  # and xtest, ttest

                for i in range(9):
                    x_train, x_test, t_train, t_test = sklearn.model_selection.train_test_split(x, t,
                                                                                                test_size=0.1)  # train_test_split
                    t_train = [i if i != 0 else -1 for i in t_train]  # changing 0 values to -1
                    t_test = [i if i != 0 else -1 for i in t_test]  # changing 0 values to -1
                    plt.figure()  # creating a new view or something
                    plt.subplot(211)  # just like a plot, but different, rows, cols
                    plt.plot(x_train[:, 0], x_train[:, 2], "b.")  # x, y, color
                    plt.title("x_train with Train_Test_Split Adaline")  # the title of our plt graph
                    plt.subplot(211)  # just like a plot, but different, rows, cols
                    plt.plot(x_test[:, 0], x_test[:, 2], "r.")  # x, y, color
                    plt.title("x_test with Train_Test_Split Adaline")  # the title of our plt graph
                    plt.tight_layout()  # to make it appear normal
                    plt.show()  # showing the graph
                    adaline(x_train, t_train, maxepochs, beta, minmse)  # calling the adaline for both x_train, t_train
                    adaline(x_test, t_test, maxepochs, beta, minmse)  # calling the adaline for both x_test, t_test

            elif userInput == 3:
                ttrain2 = [i if i != 0 else -1 for i in ttrain]  # changing 0 values to -1
                ttest2 = [i if i != 0 else -1 for i in ttest]  # changing 0 values to -1
                xtrain2 = numpy.linalg.pinv(xtrain.astype(float))  # pinv = pseudo inverse
                w = numpy.zeros(len(xtrain[0]))  # giving 0 to the w array
                predict = numpy.zeros(len(xtrain))  # initializing
                for i in range(len(w)):  # for each of the 4 elements in our w array
                    w[i] = sum(xtrain2[i, j] * ttrain2[j] for j in range(len(xtrain)))  # finding the weights
                for i in range(len(xtrain)):  # for each pattern
                    u = sum(w[j] * xtrain[i, j] for j in range(len(w)))  # initializing the stimulation
                    if u < 0:
                        predict[i] = -1  # giving the proper value
                    else:
                        predict[i] = 1  # giving the proper value
                plt.figure()  # creating new figure
                plt.plot(predict, 'ro', ttrain2,
                         'b.')  # red circles for the predict array and blue dots for the ttrain2 array
                plt.title('least square solution view')  # just the title of our plt figure
                plt.show()  # showing our plt figure
                # edw stamatise o kosmos
                xtest2 = numpy.linalg.pinv(xtest.astype(float))  # pinv = pseudo inverse
                w2 = numpy.zeros(len(xtest[0]))  # giving 0 to the w array
                predict2 = numpy.zeros(len(xtest))  # initializing
                for i in range(len(w2)):  # for each of the 4 elements in our w array
                    w2[i] = sum(xtest2[i, j] * ttest2[j] for j in range(len(xtest)))  # finding the weights
                for i in range(len(xtest)):  # for each pattern
                    u = sum(w2[j] * xtest[i, j] for j in range(len(w2)))  # initializing the stimulation
                    if u < 0:
                        print(u)
                        predict2[i] = -1  # giving the proper value
                    else:
                        print(u)
                        predict2[i] = 1  # giving the proper value
                plt.figure()  # creating new figure
                plt.plot(predict2, 'ro', ttest2,
                         'b.')  # red circles for the predict array and blue dots for the ttrain2 array
                plt.title('least square solution view')  # just the title of our plt figure
                plt.show()  # showing our plt figure
