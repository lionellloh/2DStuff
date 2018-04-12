import matplotlib.pyplot as plt

v = open("./33dot5.txt", "r")
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


plt.scatter(time_s, temp)
plt.show()

