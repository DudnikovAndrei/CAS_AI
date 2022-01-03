import numpy as np

def act(x):
    return 0 if x < 0.5 else 1

def go(house, rock, attr):
    x = np.array([house, rock, attr])
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1]
    weight1 = np.array([w11, w12])  # матрица 2x3:  3 inputs 2 output
    weight2 = np.array([-1, 1])     # вектор 1x2 2 inputs 1 output

    sum_hidden = np.dot(weight1, x)         # вычисляем семму на входах нейронов скрытого слоя
    print("Значения сумм на нейронах скрытого слоя:" + str(sum_hidden))

    out_hidden = np.array([act(x) for x in sum_hidden])
    print("знфчения на выходах нейронов скрытого слоя: " + str(out_hidden))

    sum_end = np.dot(weight2, out_hidden)
    y = act(sum_end)
    print("Выходное значение HC: " + str(y))

    return y

house = 1
rock = 1
attr = 0

res = go(house, rock, attr)
if res == 1:
    print("нравится")
else:
    print("не нравится")