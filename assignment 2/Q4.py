import pandas
import numpy as np
from sklearn import linear_model
#load the csv file
df = pandas.read_csv('heights_weights.csv')
# update the value with numbers
df['Gender'] = df['Gender'].replace(['Female'],0)
df['Gender'] = df['Gender'].replace(['Male'],1)

# convert to numpy array
data = df.to_numpy()

# taking the second and thrird row
X = data[:,1:3]
# take the first row 
y = data[:,0]

# Fit (train) the Logistic Regression classifier
clf = linear_model.LogisticRegression(C=1e40, solver='newton-cg')
fitted_model = clf.fit(X, y)

prediction_result = clf.predict([(70,180)])

if prediction_result == 1:
	print ("prediction : Male")
elif prediction_result == 0:
	print ("prediction : Female")

print ("reg score ",fitted_model.score(X, y))
print ("coef ",fitted_model.coef_)
print ("intercept_ ",fitted_model.intercept_)

# Predict
# prediction : Male
# reg score  0.9194
# coef  [[-0.49261999  0.19834042]]
# intercept_  [0.69254178]