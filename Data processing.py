import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from mlxtend.plotting import plot_linear_regression
from sklearn import neighbors, datasets
from scipy.optimize import curve_fit
import math


file = "31dot3.txt"
f = open(file, "r")

if file == "31dot3.txt":
	real_temp = 31.3

elif file == "22dot7.txt":
	real_temp = 22.7

elif file == "54dot8.txt":
	real_temp = 54.8

elif file == "10.txt":
	real_temp = 10.0

count = 0
x1 = []
x2 = []
y = []
for line in f:
	if count < 30:
		# print(type(line))
		print(line)
		count+=1
		x1.append(count)
		t = line.split(",")
		T_float = float(t[0][1:])
		x2.append(T_float)
		# y.append(22.7)

	else:
		break

# plt.scatter(x1, x2)
# plt.show()
#
# def func (x, a, b, c, d, e):
# 	return a*x**4 + b*x**3 + c*x**2 + d*x + e

# ---
# Celine's idea
# def func(x,a,b,c,d):
# 	return a*math.log(b*x,c)+d
# ---

def func(x, a, b, c):
	return a * np.log(b*x) + c

def reverse(y, a, b, c):
	return np.exp((y - c)/a)/b

# params, extras = curve_fit(func, x1[:25], x2[:25])

#Celine sucks
x1 = [26.25, 26.375, 26.437]

params = np.polyfit(np.log(np.array(x1[:25])), np.array(x2[:25]), 1)
print("params is", params)
plt.scatter(x1, x2)
x1 = np.linspace(1, 150, 150)
# plt.plot(x1, [func(x, params[0], params[1], params[2], params[3], params[4]) for x in x1])
# plt.plot(x1, [func(x, *params) for x in x1])
plt.plot(x1, [params[0]*np.log(x)+params[1] for x in x1])
plt.axhline(real_temp, label = "Real Temperature: {} celsius". format(real_temp))
plt.axvline(reverse(real_temp, *params), label = "Optimal Time Value: {} seconds".format(round(reverse(real_temp, *params),2)))
plt.axvline(reverse(real_temp+1.5, *params), label = "Limit a {} seconds".format(round(reverse(real_temp+1.5, *params),2)))
plt.axvline(reverse(real_temp-1.5, *params), label = "Limit b {} seconds".format(round(reverse(real_temp-1.5, *params),2)))
plt.title("True temperature: {}".format(real_temp))

legend = plt.legend(loc='best', shadow=True, fontsize='small')

plt.show()
# ae^-x + b

# y = np.array(y)
# print("y is", y)
# # print(x_1)
# print(x_2)

# poly = PolynomialFeatures(2,include_bias=False)
#     # print("poly", poly)
# c_data = poly.fit_transform(x_2,y)


# features = np.column_stack((x_1,x_2))
# print(features)
# #
# plt.scatter(x_1, x_2)
# plt.xlabel("time")
# plt.ylabel("temp in deg cel")
# plt.show()
#
# poly = PolynomialFeatures(2)
#
#
# # print("t = 25", linReg.predict(25))
#
# print(features.shape, y.shape)

# linReg = LinearRegression()
# linReg.fit(features, y)
# print("answer")
# print(linReg.predict([[20,43.437]]))
#54.8


############################
### CS 6
############################

### NOTE: all the xdata returned needs to be an 1D array. EG x_train[:,0] since x_train was reshaped to 2D
#
# from sklearn import linear_model
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn import neighbors, datasets
# import numpy as np
# #
# #
# #
# def multiple_linear_regression(bunchobject, x_index, y_index, order, size, seed):
#     xdata = bunchobject.data[:,x_index]
#     ydata = bunchobject.data[:,y_index]
#     xdata = xdata.reshape(-1,1)
#     ydata = ydata.reshape(-1,1)
#
#     print("x data", xdata[:5])
#     print("y data", ydata[:5])
#
#     poly = PolynomialFeatures(order,include_bias=False)
#     print("poly", poly)
#     c_data = poly.fit_transform(xdata,ydata)
#     print("c _data", c_data)
#
#     x_train, x_test, y_train, y_test = train_test_split(c_data , ydata , test_size=size , random_state=seed)
#     regr = linear_model.LinearRegression()
#     regr.fit(x_train, y_train)
#     y_pred = regr.predict(x_test)
#
#     results = {}
#     results["coefficients"] = np.array(regr.coef_)
#     results["intercept"] = np.array(regr.intercept_)
#     results["mean squared error"] = mean_squared_error(y_test, y_pred)
#     results["r2 score"] = r2_score(y_test,y_pred)
#
#     return x_train[:,0], y_train, x_test[:,0], y_pred, results
#
# bunchobject = datasets.load_breast_cancer()
# print(bunchobject)
# x_train, y_train, x_test, y_pred, results = multiple_linear_regression(bunchobject, 0 , 3 , 4 , 0.4 , 2752 )
# print("RESULTS", results)
#
# plt.scatter(x_train, y_train)
# plt.show()
# print(plot_linear_regression(x_train, y_train, x_test, y_pred, bunchobject.feature_names[0], bunchobject.feature_names[3]))
# plt.show()

