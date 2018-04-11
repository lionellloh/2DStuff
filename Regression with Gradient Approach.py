import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from mlxtend.plotting import plot_linear_regression
from sklearn import neighbors, datasets
from scipy.optimize import curve_fit
import math


def func(x, m, c):
		return m*x+c

files = ["10.txt", "22dot7.txt", "31dot3.txt", "46.txt", "54dot8.txt"]
collated = {}
for file in files:

	f = open(file, "r")

	if file == "31dot3.txt":
		real_temp = 31.3

	elif file == "22dot7.txt":
		real_temp = 22.7

	elif file == "54dot8.txt":
		real_temp = 54.8

	elif file == "10.txt":
		real_temp = 10.0

	elif file == "46.txt":
		real_temp = 46.0

	count = 0
	time_s = []
	temp = []
	for line in f:
		if count < 30:
			# print(type(line))
			# print(line)
			count+=1
			time_s.append(count)
			t = line.split(",")
			# print(t)
			T_float = float(t[0][1:])
			temp.append(T_float)



	collated[file] = [real_temp]
	for i in range(5,30,5):
		params, extras = curve_fit(func, time_s[:i], temp[:i])
		# Plot the Temperature against time graph
		# plt.scatter(time_s, temp)
		# plt.plot(time_s, [func(x, *params) for x in time_s])
		print("for {} file,  {} data points: m = {}, c = {}".format(file, i, params[0], params[1]))
		collated[file].append(params[0])
	# plt.figure(file)
	# plt.show()

print(collated)

y = [10, 22.7, 31.3, 46, 54.8]

# Using curve fitting instead of LinearRegression
for i in range(5):
	X = [collated[file][i+1] for file in files]
	print("The list of gradients is", X)
	lm = LinearRegression()

	# model = lm.fit(X, y)
	# print("for {} data points, score is {}".format(i*5,lm.score(X,y)))

	params, extras = curve_fit(func, X, y)
	plt.scatter(X, y)
	plt.plot(X, [func(x, *params) for x in X])
	print("m is {}, x is {} using {} data points".format(params[0], params[1], (i+1)*5))
plt.show()


# JUST TESTING THE PREDICTION


