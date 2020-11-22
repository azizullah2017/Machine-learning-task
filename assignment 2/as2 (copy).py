import numpy as np
from sklearn import linear_model
# from sklearn.linear_model
file1 = open('train.txt', 'r') 
Lines = file1.readlines() 
# Density determined from underwater weighing
# Percent body fat from Siri's (1956) equation
header = "Density , Percent , Age , Weight, Height, Neck circumference, Chest circumference, Abdomen 2 circumference, Hip circumference , Thigh circumference,  Knee circumference, Ankle circumference ,Biceps (extended) circumference, Forearm circumference ,Wrist circumference"
 # [
 # "Density underwater weighing", x
 # "Percent body fat",
 # "Age",
 # "Weight",
 # "Neck circumference",
 # "Chest circumference",
 # "Abdomen 2 circumference",
 # "Hip circumference", 
 # "Thigh circumference", 
 # "Knee circumference", 
 # "Ankle circumference", 
 # "Biceps (extended) circumference", 
 # "Forearm circumference", 
 # "Wrist circumference"]
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
# arr = np.array([14 ,21, 13, 56, 12])
# np.savetxt("foo.csv", np.dstack((np.arange(1, arr.size+1),arr))[0],"%d,%d",header="Id,Values")
# np.savetxt("foo.csv", np.dstack((np.arange(1, data.size+1),data))[0],"%d,%d",header=header)
# np.savetxt("foo.csv", data, delimiter=",")
#find the shape
# print (data)
y = data[:,1]
# print (y)
X = data[:,2:15]
# print (X)
# take the first row 
# y = data[:,1]

# y=y.astype('int')
# clf = linear_model.LogisticRegression(C=1e40, solver='newton-cg')
# fitted_model = clf.fit(X, y)

fitted_model = linear_model.LinearRegression().fit(X, y)
print ("reg score ",fitted_model.score(X, y))
print ("coef ",fitted_model.coef_)
print ("intercept_ ",fitted_model.intercept_)
# example = [23,154.25,67.75,36.2,93.1,85.2,94.5,59.0,37.3,21.9,32.0,27.4,17.1]
example = [27,133.25,64.75,36.4,93.5,73.9,88.5,50.1,34.5,21.3,30.5,27.9,17.2]
print ("predict : ",fitted_model.predict(np.array([example])))

# import numpy as np
# from sklearn.linear_model import LinearRegression
# X = np.array([[1, 1], 
# 	[1, 2], 
# 	[2, 2],
# 	[2, 3]])
# print(X)
# # y = 1 * x_0 + 2 * x_1 + 3
# y = np.dot(X, np.array([1, 2])) + 3
# print("y",y)
# reg = LinearRegression().fit(X, y)
# print ("reg score ",reg.score(X, y))
# print ("coef ",reg.coef_)
# print ("intercept_ ",reg.intercept_)
# print ("predict 3,5 : ",reg.predict(np.array([[2, 3]])))

# In the case of Linear Regression, the outcome is continuous 
# while in the case of Logistic Regression outcome is discrete (not continuous)

# Linear regression is used for regression or to predict continuous values 
# whereas logistic regression can be used both in classification and regression problems 
# but it is widely used as classification algorithm.
# Regression models aim to project value based on independent features.

# You can only multiply matrices if the number of columns of the first matrix is the same as the number of rows as the second matrix.
# For example, say you want to multiply A x B. If A is a 3x1 matrix, B has to be a 1xY matrix (Y can be any number), because A only has 1 column.


# import numpy as np
# data = [[1,2,3],[1,2,3]]
# d = np.array(data)

# print (d.shape)
# d = np.append(d, a, axis=1)
# print (d)
