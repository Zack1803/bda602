import numpy

nd_array_1 = numpy.array([[1, 2, 3], [4, 5, 6]])  # Create a 2x3 matrix
print(nd_array_1)
print(nd_array_1.shape)
print(nd_array_1.sum())
print(nd_array_1.sum(axis=0))  # Sum by columns

print(nd_array_1.sum(axis=1))  # Sum by rows

print(nd_array_1.mean())  # Mean of everything in the array
print(nd_array_1.mean(axis=0))  # Mean by columns
print(nd_array_1.argmax())  # location of where the MAX value is
print(type(nd_array_1))
print(nd_array_1)
print(nd_array_1.tolist())
empty_matrix = numpy.zeros(shape=(4, 10))  # 4x10 empty matrix
print(empty_matrix)
