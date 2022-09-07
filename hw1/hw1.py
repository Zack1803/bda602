import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import datasets
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def modelling(x_val, y_val):
    pipeline = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        x_val, y_val, test_size=0.3, random_state=42
    )
    print("Original Size of X", len(x_val))
    print("Original size of y", len(y_val))
    print("X-Train Size", len(X_train))
    print("y-Train Size", len(y_train))
    print("X-Test Size", len(X_test))
    print("y-Test Size", len(y_test))

    pipeline.fit(X_train, y_train)
    probability = pipeline.predict_proba(X_test)
    prediction = pipeline.predict(X_test)
    print(f"Probability: {probability}")
    print(f"Predictions: {prediction}")
    print(pipeline.score(X_test, y_test))

    BoostPipe = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=100)),
        ]
    )
    BoostPipe.fit(X, y)
    probability_boost = BoostPipe.predict_proba(X_test)
    prediction_boost = BoostPipe.predict(X_test)
    print(f"Probability: {probability_boost}")
    print(f"Predictions: {prediction_boost}")
    print(pipeline.score(X_test, y_test))


df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = pd.Series(iris.target)
print(df.head(5))
summ("sepal length", df["sepal length (cm)"])
summ("sepal length", df["sepal width (cm)"])
summ("sepal length", df["sepal width (cm)"])
summ("sepal length", df["petal width (cm)"])
summ("sepal length", df["target"])
plot()
X = df.loc[:, df.columns != "target"]
y = df["target"]
modelling(X, y)
