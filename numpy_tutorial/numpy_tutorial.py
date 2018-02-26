import numpy as np

"""
Array
"""

a = np.array([1, 2, 3])
print(type(a))
print(a.shape)
print(a[0], a[1], a[2])
a[0] = 5
print(a)

b = np.array([[1,2,3], [4,5,6]])
print(type(a))
print(b)
print(b.shape)
print(b[0,0], b[0,1], b[1])
print(type(b[1]))
print(type(b[1][0]))

print(b[1,0])
print(b[1][0])


a = np.zeros((2, 2))   # Create an array of all zeros
print(a)

b = np.ones((1,2))
print(b)

c = np.full((2,2), 7)
print(c)

d = 5 * np.eye(2)
print(d)

e = np.random.random((2,2))   # 均匀分布
print(e)


"""
Array index
"""
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)
b = a[:2, 1:3]
print(b)
print(type(b))

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(a[0, 1])
b[0,0] = 70
print(a[0,1])


a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[[1], :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)
print(col_r2, col_r2.shape)

a = np.array([[1,2], [3, 4], [5, 6]])
print(a)
print(type(a[[0,1,2],[0,1,0]]))
print((a[[0,1,2],[0,1,0]]))
print(type(a[1,1]))
# print(a[[0, 1, 2], [0, 1, 0]])等价于下式
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))

a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(a)
b = np.array([0,2,0,1])
print(np.arange(4))

print(a[np.arange(4), [0,2,0,1]])
print(a[np.arange(4), np.array([0,2,0,1])])

# mutate the ndarray
a[np.arange(4), b] += 10
print(a)
a[list(range(4)), [0,2,0,1]] += 10
print(a)


"""
Boolean array indexing
"""
a = np.array([[1,2], [3, 4], [5, 6]])
bool_idx = a>2
print(bool_idx)
print(type(bool_idx))
print(a[bool_idx])
print(a[a>2])

"""
Datatypes:numpy中的元素要求类型相同
"""
x = np.array([1,2])
print(x.shape)
print(type(x[0]))
print(x.dtype)

x = np.array([1.,2.,3.])
print(x.dtype)

x = np.array([1,2.,'a'])
print(x.dtype)
print(type(x[[0]]))
print(x[[0]])

x = np.array([1,2], dtype=np.float64)
print(x.dtype)
print(x)


"""
Array math
"""
print('*'*100)
x = np.array([[1,2], [3,4]], dtype=np.float64)
y = np.array([[5,6], [7,8]], dtype=np.float64)
print(x + y)
print(np.add(x, y))

print(x-y)
print(np.subtract(x,y))


print(x * y)
print(np.multiply(x, y))

print(x / y)
print(np.divide(x, y))


print(np.sqrt(x))
print(x ** 2)


# 矩阵相乘，用dot
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

print(x.dot(y))
print(np.dot(x, y))

v = np.array([9, 10])
w = np.array([11, 12])

# Inner product of vectors, ndarray中向量是1维的，和二维矩阵相乘，还会得到向量(2,)
print(v.dot(w))
print(np.dot(v, w))


# Matrix product
print(x.shape)
print(v.shape)
print(x.dot(v))
print(np.dot(v, x))
v2 = np.array([[9, 10]])
print(v2.dot(x))
v3 =  np.array([[9], [10]])
print(x.dot(v3))  # 这里x.shape=(2,2),v3.shape(2,1)，相乘后得到的matrix,shape=(2,1)


x = np.array([[1,2], [3,4]])
print(np.sum(x))  # 所有元素求和
print(np.sum(x, axis=0))  # 沿着第一维的方向（即第1行，第二行，第三行，以此类推）求和，求和会降维
print(np.sum(x, axis=1))  #


z = np.array([x,2*x])
print(z)
print(np.sum(z, axis=0))

x = np.array([[1,2], [3,4]])
print(x.T)

# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1,2,3,4])
print(v.T)

"""
Broadcasting: http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc
"""
print('*'*100)
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1,0,1])
y = np.empty_like(x)   # 其中empty_like()只分配数组所使用的内存，不对数组元素进行初始化操作，因此它的运行速度是最快的

print(x)
print(v)
print(y)

for i in range(4):
    y[i, :] = x[i, :] + v

print(y)

vv = np.tile(v, (4, 1))
print(vv)

y = x + vv
print(y)

y = x + v
print(y)

a = np.array([1,2,3])
b = 2
print(a*b)  #  广播


x = np.arange(4)
xx = x.reshape(4,1)
y = np.ones(5)
z = np.ones([3, 4])

print(xx.shape)  # (4, 1)
print(y.shape)   # (   5,)
print((xx+y).shape)  # (4, 5)
print(xx + y)  # xx复制5列, y复制4行


print(x.shape, x)  # (4,)
print(z.shape, z)  # (3, 4)
print(x + z)  # x复制3行, (3,4)

a = np.array([0.0, 10.0, 20.0, 30.0])
b = np.array([1.0, 2.0, 3.0])
print(a[:, np.newaxis])
print(a[np.newaxis, :])
print(a[:, np.newaxis] / b)

# A Practical Example: Vector Quantization.
observation = np.array([111., 188])
codes = np.array([[102.0, 203.0],
                   [132.0, 193.0],
                    [45.0, 155.0],
                    [57.0, 173.0]])
diff = codes - observation
print(diff)
dist = np.sqrt(np.sum(diff**2, axis=1))
print(dist)
print(np.argmin(dist))


# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)

print(np.reshape(v, (3, 1)) * w)
print(v[:, np.newaxis] * w)

x = np.array([[1,2,3], [4,5,6]])
print(x + v)

# 把w加到x的每列上，
print(x)
# 这个得先转置x，在广播w，在转置
print((x.T + w).T)
# 或者把w变成（2,1)，再与x广播相加
print(x + np.reshape(w, (2,1)))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print(x * 2)






