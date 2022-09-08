import pandas as pd

iris = pd.read_fwf("iris.data")
print(iris.head(5))
head = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "target",
]
iris = iris.reset_index()
iris.columns = head
print(iris.head(5))
