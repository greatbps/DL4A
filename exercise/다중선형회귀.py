import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1 = [i[0] for i in data]
x2 = [i[1] for i in data]
y = [i[2] for i in data]

ax = plt.axes(projection='3d')
ax.set_xlabel('study_hours')
ax.set_ylabel('private_class')
ax.set_zlabel('Score')

ax.dist = 11
ax.scatter(x1, x2, y)

plt.show()

x1_data = np.array(x1)
x2_data = np.array(x2)
y_data = np.array(y)

a1 = 0
a2 = 0
b = 0

Ir = 0.02
epochs = 2001

for i in range(epochs):
    y_pred = a1 * x1_data + a2 * x2_data + b
    error = y_data - y_pred
    a1_diff = -(2/len(x1_data)) * sum(x1_data * (error))
    a2_diff = -(2/len(x2_data)) * sum(x2_data * (error))
    b_diff = -(2/len(x1_data)) * sum(y_data - y_pred)

    a1 = a1 - Ir * a1_diff
    a2 = a2 - Ir * a2_diff
    b = b - Ir * b_diff

    if i % 100 == 0:
        print("epoch=%.f, 기울기1=%.04f, 기울기2=%.04f, 절편=%.04f" % (i, a1, a2, b))

        



