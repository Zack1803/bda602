import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import datasets

iris = datasets.load_iris()
# X = iris.data[:, :2]  # we only take the first two features.
# y = iris.target
# print(X)
# print(y)

# a simple function to compute summary of data.


def summ(s, attr):
    mean_value = np.mean(attr)
    max_value = np.max(attr)
    min_value = np.min(attr)
    quartiles_value = np.quantile(attr, [0.25, 0.5, 0.75])
    print("Attribute", "Mean", "Max", "Min", "Quartile")
    print(s, mean_value, max_value, min_value, quartiles_value)

    # describe = pd.DataFrame([{"Mean":mean_value,"Max":max_value,"Min":min_value,"Quartile":quartiles_value}])
    # print(describe)


def plot():
    sctplot = px.scatter(
        df, x="sepal length (cm)", y="petal length (cm)", color="target"
    )
    sctplot.show()

    fig = px.scatter(
        df, x="sepal width (cm)", y="sepal length (cm)", color="petal length (cm)"
    )
    fig.show()

    # Boxplot
    boxplt = px.box(
        df, x="target", y="petal length (cm)", color="target", title="Flower Type"
    )
    boxplt.show()

    # Violin plot
    violplt = px.violin(
        df,
        x="target",
        y="petal length (cm)",
        color="target",
        title="Flower Type",
    )
    violplt.show()

    # Pair plots
    pairs = px.scatter_matrix(
        df,
        color="target",
        title="Pair Plot",
        labels=df.columns,
    )
    pairs.update_traces(diagonal_visible=False)
    pairs.show()

    # Heatmap
    heat = px.imshow(df)
    heat.show()


df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = pd.Series(iris.target)
print(df.head(5))
summ("sepal length", df["sepal length (cm)"])
summ("sepal length", df["sepal width (cm)"])
summ("sepal length", df["sepal width (cm)"])
summ("sepal length", df["petal width (cm)"])
summ("sepal length", df["target"])
plot()
