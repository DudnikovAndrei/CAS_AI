import  numpy as np
import matplotlib.pyplot as plt

N = 100        # число экспериментов
sigma = 3       # стардартное отклонение наблюдаемых значений
k = 0.5           # теоретическое значение параметра k
b = 2           # теоретическое значение параметра b

f = np.array([k*z+b for z in range(N)])
y = f + np.random.normal(0, sigma, N)
x = np.array(range(N))

print(x)

print(y)

mx = x.sum() / N
print("mx: ", mx)
my = y.sum() / N
print("my: ", my)

a2 = np.dot(x.T, x) / N
print("a2: ", a2)
a11 = np.dot(x.T, y) / N
print("a11: ", a11)

kk = (a11 - mx * my) / (a2 - mx**2)
bb = my - kk * mx

ff =  np.array([kk*z+bb for z in range(N)])

plt.plot(f)
plt.plot(ff, c='red')
plt.scatter(x, y, s=2, c='red')
plt.grid(True)
plt.show()