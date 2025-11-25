import numpy as np

x = [2, 4, 6, 8]
x = np.array(x)

print("EXERCICIO 1: \n")
mean = x.mean()
print(f"Media: {mean}")
print(f"Cada desvio: {x - mean}")
print(f"Cada desvio ao quadrado: {(x - mean) ** 2}")
print(f"Variancia: {x.var()}")

print("\nEXERCICIO 2:\n")
v = np.array([3, 4])
print(f"Norma L2: {np.linalg.norm(v)}")
v2 = np.array([1, -2, 2])
print(f"Norma L1: {np.linalg.norm(v2, ord=1)}")
print(f"Norma L2: {np.linalg.norm(v2)}")
print(f"Norma L-inf: {np.linalg.norm(v2, ord=np.inf)}")


print("\nEXERCICIO 3:\n")
a = np.array([1, 2, 3])
b = np.array([4, 0, -2])

print(f"Dot: {np.dot(a, b)}")


print("\nEXERCICIO 4:\n")
X = [1, 2, 3, 4]
Y = [1, 3, 2, 4]

x_mean = np.mean(X)
y_mean = np.mean(Y)

print(f"MEDIA X: {x_mean}\nMEDIA Y: {y_mean}")
print((X - x_mean) * (Y - y_mean))
print(f"Covariancia: {np.sum(np.cov(X, Y))}")


print("\nEXERCICIO 5:\n")

A = [0, 1, 2, 3]
B = [6, 4, 2, 0]

print(f"Covariancia: {np.sum(np.cov(A, B))}")


print("\nEXERCICIO 6:\n")

A = [0, 1, 2, 3]
B = [6, 4, 2, 0]

print(f"Covariancia: {np.cov(A, B)}")

print("\nEXERCICIO 7:\n") ### Revisar

X = np.array([[1, 2],
     [3, 6],
     [5, 10]])
    
x_mean = np.mean(X[:,0])
y_mean = np.mean(X[:,1])

x = X[:,0] - x_mean
y = X[:,1] - y_mean

print(x, y)
print(np.mean(x*x))
print(np.mean(y*y))
print(np.mean(x*y))

s = [
    [np.mean(x*x), np.mean(x*y)],
    [np.mean(x*y), np.mean(y*y)]
]

print(s)

print("\nEXERCICIO 8:\n")

A = np.array([[1, 2, 3],
     [4, 5, 6]])

print(f"TRANSPOSTA: \n{A.T}")


print("\nEXERCICIO 9:\n")

A = np.array([[1, 2],
     [3, 4]])

B = np.array([[2, 0],
     [1, 2]])

print(np.dot(A, B))
print()
print(np.dot(B, A))


print("\nEXERCICIO 9:\n")

M = np.array([[1, -1],
     [2,  2]])

print("Norma de matriz (Frobenius) - ||M||F = sqrt(sum de cada elemento ao quadrado) :")
print(np.sqrt(np.sum(np.hstack(M) ** 2)))


print("\nEXERCICIO 10:\n")

X = np.arange(16).reshape(2,2,2,2)
print(X)
print()
X2 = np.transpose(X, (3,2,1,0))
print(X2)

print("\nEXERCICIO 11:\n")

v = np.array([1, 2, 3])
M = np.array([[10, 20, 30],
              [40, 50, 60]])

print(M + v)


print("\nEXERCICIO 11:\n")

A = np.array([[1,2,3],
              [4,5,6]])

print(np.sum(A, axis=1))
print(np.sum(A, axis=0))