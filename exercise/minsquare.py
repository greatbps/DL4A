import numpy as np

x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

mx = np.mean(x)
my = np.mean(y)

print("x의 평균값", mx)
print("y의 평균값", my)

divisor = sum([i - mx ** 2 for i in x])

def top(x, mx, y, my):
    d = 0

    for i in range(len(x)) :
        d += (x[i] - mx) * (y[i] - my)
    return d
dividend = top(x, mx, y, my)

a = dividend / divisor

print("분모", divisor)
print("분자", dividend)
b = my - (mx* a)

print("기울기 a=", a)
print("y절편 b=", b)