import pandas
import numpy
import 

data = pandas.read_csv('iris.data', header=None).values  # reading from the csv file
flowerInfo = {
    'numberOfPatterns': numpy.shape(data)[0],  # dictionary array for patterns
    'numberOfAttributes': numpy.shape(data)[1]  # and attributes
}
x = data[:, :4]  # giving the values of the flowers without the name
names = data[:, 4]  # giving the names without the values
x[:, 0].plot(kind='bar')