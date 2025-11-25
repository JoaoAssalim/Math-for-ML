import numpy as np

# Scalars
scalar = np.array(3)
print(f"SCALAR: {scalar}\nSHAPE: {scalar.shape}")
print(f"SQRT: {np.sqrt(scalar)}")
print(f"ABS: {np.abs(scalar)}")
print(f"LOG: {np.log(scalar)}")

print()

# Vectors

vector = np.array([1, 2, 3])
print(f"Vector Shape: {vector.shape}")
print(f"Vector + scalar: {vector + scalar}")
print(f"Vector * scalar: {vector * scalar}")
print(f"Vector + Vector: {vector + vector}")
print(f"Vector * Vector: {vector * vector}")
print(f"SQRT: {np.sqrt(vector)}")
print(f"Norma: {np.linalg.norm(vector)}") # Representa a distancia do vetor até a origem
print(f"Manual Norma: {np.sqrt(np.sum(vector ** 2))}") # Raiz quadrada da soma dos elementos ao quadrado
print(f"Produto interno: {np.dot(vector, vector)}") # o quanto um vetor aponta na direção do outro (usado para medir distancia de vetores)

print()

# Matrices

matrix = np.array([[1, 2], [4, 5]])
print(f"Matrix shape: {matrix.shape}")
print(f"Matrix Transposta: \n{matrix.T}\n")
print(f"Multiplicação matricial: \n{np.dot(matrix, matrix)}\n") # para multiplicar, o numero de colunas de X deve ser igual ao numero de colunas de Y

matrix_2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Multiplicação pela transposta: \n{np.dot(matrix_2, matrix_2.T)}\n")

# Pq multiplicar pela transposta?
# A·Aᵀ → Matriz de correlação/distâncias entre linhas
# Aᵀ·A → Matriz de correlação/distâncias entre colunas

# Você obtém uma matriz onde:
# elemento (0,0) = norma ao quadrado da primeira linha
# elemento (1,1) = norma ao quadrado da segunda linha
# elemento (0,1) = produto interno entre linha 1 e linha 2

print(f"Determinante: {np.linalg.det(matrix)}")
print(f"Matrix inversa: \n{np.linalg.inv(matrix)}\n")
print(f"Traço: {np.trace(matrix)}") # soma da diagonal principal

vals, vecs = np.linalg.eig(matrix)
print(f"AUTOVALOR: \n{vals}\n") # É o quanto a matriz estica ou encurta
print(f"AUTOVETOR: \n{vecs}\n") # Vetor especial que sempre mantem a mesma direção


X = np.array([[4, 3], [1, -5]])
y = np.array([6, 8])

print(f"SOLVED: {np.linalg.solve(X, y)}")

print()

# Tensores

X = np.random.randn(32, 3, 224, 224)
print(f"SHAPE: {X.shape}")
print(f"NDIM: {X.ndim}")
print("Indexação multi-dimensional: {X[0, :, :, :]}")

print()

# Outros metodos

X = np.array([1, 2, 3, 4])
print(X.reshape(2, 2)) # muda a dimensão sem alterar os dados

print()

X = np.random.randn(1, 10, 1)
print(X) # (1, 10, 1)
print()
print(X.squeeze()) # Remove dimensões de tamanho 1 (10,)

print()

x = np.array([1,2,3]) 
x2 = np.expand_dims(x, axis=0) # adiciona uma nova dimensão do tamanho 1 row
print(x2)
x2 = np.expand_dims(x, axis=1) # adiciona uma nova dimensão do tamanho 1 column
print(x2)


print()

print(np.transpose(np.array([[1, 2], [3, 4]])))

X = np.random.randn(32, 3, 224, 224)
print(X.shape)
X2 = np.transpose(X, (0, 2, 3, 1))
print(X2.shape)

print("\n\n\n")
X = np.random.randn(2, 2, 2, 2)
print(X)
print(X.shape)
X2 = np.transpose(X, (3, 2, 1, 0))
print()
print(X2)
print(X2.shape)


print("\n\n\n")
X = np.random.randn(2, 2)
print(X)
print(X.shape)
X2 = np.transpose(X, (1, 0))
print()
print(X2)
print(X2.shape)


print()

# broadcasting

A = np.ones((3,3))
b = np.array([1,2,3])

print(b + A)

print()

print(np.concatenate(([1,2], [2, 3])))
print(np.stack(([1,2], [2, 3])))
print(np.split(np.array([1,2,3,4,5,6]), 2))
print(np.argmax(np.array([1,2,23,4])))
print(np.argmin(np.array([1,2,23,4])))
print(np.mean(np.array([1,2,23,4])))
print(np.std(np.array([1,2,23,4])))


print(np.array([1,2,23,4]).mean())
print(np.array([1,2,23,4]).var())

print(np.cov(X, rowvar=False))