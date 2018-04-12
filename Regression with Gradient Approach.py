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

# I will try to validate model with 39dot 6
files = ["./Data_new/10dot7.txt", "./Data_new/14dot8.txt", "./Data_new/16dot8.txt", "./Data_new/21dot1.txt",
		 "./Data_new/24dot7.txt","./Data_new/30dot4.txt","./Data_new/35dot0.txt", "./Data_new/50dot8.txt",
		 "./Data_new/59dot9.txt"]

values = [10.7, 14.8, 16.8, 21.1, 24.7, 30.4, 35.0, 50.8, 59.9]

collated = {}

for i in range(len(files)):

	f = open(files[i], "r")

	real_temp = values[i]

# Below is the code to parse and store the file data in 2 lists
	count = 0
	time_s = []
	temp = []
	for line in f:
		if count < 30:
			# print(type(line))
			# print(line)
			count+=1
			time_s.append(count)
			t = line.split(" ")
			# print(t)
			T_float = float(t[0])
			temp.append(T_float)

	collated[files[i]] = [real_temp]
	print(collated)
	for j in range(5,30,5):
		params, extras = curve_fit(func, time_s[:j], temp[:j])
		# Plot the Temperature against time graph
		# plt.scatter(time_s, temp)
		# plt.plot(time_s, [func(x, *params) for x in time_s])
		print("for {} file,  {} data points: m = {}, c = {}".format(files[i], j, params[0], params[1]))
		collated[files[i]].append(params[0])
	# plt.figure(file)
	# plt.show()

print(collated)


# Using curve fitting instead of LinearRegression
for k in range(5):
	print(k)
	X = [collated[files[i]][k+1] for i in range(len(files))]
	print("The list of gradients is", X)
	# lm = LinearRegression()

	# model = lm.fit(X, y)
	# print("for {} data points, score is {}".format(i*5,lm.score(X,y)))

	params, extras = curve_fit(func, X, values)
	plt.scatter(X, values)
	plt.plot(X, [func(x, *params) for x in X])
	print("m is {}, c is {} using {} data points".format(params[0], params[1], (k+1)*5))
plt.show()




# JUST TESTING THE PREDICTION


v = open("./Data_new/33dot5.txt", "r")
count = 0
time_s = []
temp = []
for line in v:
	if count < 30:

		count+=1
		time_s.append(count)
		t = line.split(" ")
		# print(t)
		T_float = float(t[0])
		temp.append(T_float)
print("temp is", temp)

def find_gradient(temp):
	def func(x, m, c):
		return m*x + c

	time = np.linspace(1,30, 30)
	print("time is", len(time))
	params, extras = curve_fit(func, time_s[:20], temp[:20])
	return params[0]

grad = (find_gradient(temp))
print(grad)
model = LinearRegression()
model.fit(time_s[:20], temp[:20])
ans = model.predict(grad)
print("model is", ans)
# m is 32.80767290672981, c is 27.792797546632755 using 20 data points

temp = 32.80767290672981 * grad + 27.792797546632755

print("temperature is", temp)
