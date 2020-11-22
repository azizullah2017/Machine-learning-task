import numpy as np
from sklearn import linear_model
file1 = open('train.txt', 'r') 
Lines = file1.readlines() 
data = []
# Strips the newline character 
for line in Lines:
	# slpit list of string and remove the newline tag
	values = line.strip().split("  ")
	# removing empty values from list
	values = list(filter(None, values))
	# converting string to float values
	for i in range(0, len(values)):
		values[i] = float(values[i])
	# list of data
	data.append(values)
# convert data to numpy array
data = np.array(data)

# y is Percent body fat
y = data[:,1]

# excluding the body density
X = data[:,2:15]

xo = []
# create row of 252 with single col
d = np.ones(shape= (252,1),dtype = np.int32)
# print ()
for x in range(252):
	xo.append([1])
# add xo to features
X= np.append(X, xo, axis=1)

# calculate W0 to W13

X_transpose = np.transpose(X)
result_inverse =(np.linalg.inv((X_transpose.dot(X))))
xIntoY = X_transpose.dot(y)
Weights = (result_inverse.dot(xIntoY))
print (Weights)

# ############## 
#   out put   #
# ############## 

# [ 6.20786464e-02 -8.84446759e-02 -6.95904296e-02 -4.70600014e-01
#  -2.38641465e-02  9.54773458e-01 -2.07541123e-01  2.36099845e-01
#   1.52812146e-02  1.73995368e-01  1.81602416e-01  4.52024914e-01
#  -1.62063910e+00 -1.81884851e+01]

e = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
f = e.reshape(3,4) # coverted to col and row of 3 by 4
# e[5:] #change last five
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

a = np.array([1,2,3])
b= np.vstack([a,a,a])
c= np.hstack([a,a,a])
g = np.concatenate([a,a])

# print (g)
import matplotlib.pyplot as plt
x = np.linspace(0,2,10)
# 10 elements with value range from 0 to 2
# plt.plot(x,'o-') # point notation
# plt.show()

# basic scatter plot
# x = np.array([1, 2, 3])
# y = np.array([2, 4, 6])
# plt.scatter(x,y)
# plt.xlabel('some x axis')
# plt.ylabel('some y axis')
# plt.show()


# plt.plot(x,x,'o-',label='linear')
# plt.plot(x,x**2,'x-',label='Quadratic') 
# plt.legend(loc='best')
# plt.title('linear vs Quadratic')
# plt.xlabel('input')
# plt.ylabel('out')
# plt.show()

samples = np.random.normal(loc=0.1,scale=0.5,size=1000)
# print (samples.dtype)
# print (samples[:30])
# plt.hist(samples, bins=50) # bins number of bars
# plt.show()


samples_1 = np.random.normal(loc=1,scale=0.5,size=1000)
samples_2 = np.random.standard_t(df=10,size=1000)

# bins = np.linspace(-3,3,50)
# _ =  plt.hist(samples_1, bins=bins, alpha = 0.5, label="samples 1")
# _ =  plt.hist(samples_2, bins=bins, alpha = 0.5, label="samples 2")
# plt.legend('upper left')

plt.scatter(samples_1,samples_2, alpha=0.1)
plt.show()

# print (X.dot(X_transpose))
# # y=y.astype('int')
# # clf = linear_model.LogisticRegression(C=1e40, solver='newton-cg')
# # fitted_model = clf.fit(X, y)

# fitted_model = linear_model.LinearRegression().fit(X, y)
# print ("reg score ",fitted_model.score(X, y))
# print ("coef ",fitted_model.coef_)
# print ("intercept_ ",fitted_model.intercept_)
# # example = [23,154.25,67.75,36.2,93.1,85.2,94.5,59.0,37.3,21.9,32.0,27.4,17.1]

# # testing on model
# example = [27,133.25,64.75,36.4,93.5,73.9,88.5,50.1,34.5,21.3,30.5,27.9,17.2]
# print ("predict : ",fitted_model.predict(np.array([example])))