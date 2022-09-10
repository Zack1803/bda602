import pandas as pd
import plotly.express as px
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def plot(df):
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
    heat = px.imshow(
        df[
            [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)",
            ]
        ]
    )
    heat.show()


def Model_metrics(X_orig, y, pipeline):
    pipeline.fit(X_orig, y)
    probability = pipeline.predict_proba(X_orig)
    prediction = pipeline.predict(X_orig)
    print(f"Probability: {probability}")
    print(f"Predictions: {prediction}")


def modelling(df):
    X_orig = df.loc[:, df.columns != "target"]
    y = df["target"]
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(df["target"])
    pipeline = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("OneHotEncoder", OneHotEncoder()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )
    Model_metrics(X_orig, y, pipeline)
    BoostPipe = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=100)),
        ]
    )
    Model_metrics(X_orig, y, BoostPipe)


iris = pd.read_csv("iris.data", header=None)
head = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "target",
]
iris.columns = head
print(iris.shape)
print(iris.head())
df = iris
plot(df)
modelling(df)
