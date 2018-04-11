import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from mlxtend.plotting import plot_linear_regression
from sklearn import neighbors, datasets
from scipy.optimize import curve_fit
import math

#
def func(x, m, c):
		return m*x+c

files = ["./Data_new/10dot7.txt", "./Data_new/14dot8.txt", "./Data_new/16dot8.txt", "./Data_new/21dot1.txt",
		 "./Data_new/24dot7.txt","./Data_new/30dot4.txt","./Data_new/35dot0.txt", "./Data_new/50dot8.txt",
		 "./Data_new/59dot9.txt"]
collated = {}
for file in files:

	f = open(file, "r")

	if file == "10dot7.txt":
		real_temp = 10.7

	elif file == "14dot8.txt":
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
	print("m is {}, c is {} using {} data points".format(params[0], params[1], (i+1)*5))
plt.show()


# JUST TESTING THE PREDICTION


# v = open("./Data_new/59dot9.txt", "r")
# count = 0
# time_s = []
# temp = []
# for line in v:
# 	if count < 30:
#
# 		count+=1
# 		time_s.append(count)
# 		t = line.split(" ")
# 		# print(t)
# 		T_float = float(t[0])
# 		temp.append(T_float)
# print(temp)
#
# def find_gradient(temp):
# 	def func(x, m, c):
# 		return m*x + c
#
# 	time = np.linspace(1,30, 30)
# 	print("time is", time)
# 	params, extras = curve_fit(func, time[:20], temp[:20])
# 	return params[0]
#
# print(find_gradient(temp))
#
# # m is 26.847307126026752, x is 28.905414702084453 using 20 data points
#
# temp = 26.847307126026752 * 0.969417293233 + 28.905414702084453
#
# print(temp)
