import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodelsgit .api as sm
from plotly import figure_factory as ff
from plotly import graph_objects as go
from scipy import stats
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

# Diabetes


# diabetes = datasets.load_diabetes(as_frame=True)
# df = diabetes['frame']
# predictors = ['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6']
# response = ['target']

# Iris
df = pd.read_csv("iris.csv")
df.head()
predictors = ["Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
response = ["Species"]

import sys

import statsmodels.api
from plotly import express as px
from sklearn import datasets


def lr(df):
    X = df.iloc[::-1]
    y = df.iloc[:-1]
    for idx, column in enumerate(X.T):
        feature_name = diabetes.feature_names[idx]
        predictor = statsmodels.api.add_constant(column)
        linear_regression_model = statsmodels.api.OLS(y, predictor)
        linear_regression_model_fitted = linear_regression_model.fit()
        print(f"Variable: {feature_name}")
        print(linear_regression_model_fitted.summary())
        # Get the stats
        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

        # Plot the figure
        fig = px.scatter(x=column, y=y, trendline="ols")
        fig.update_layout(
            title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="y",
        )
        fig.show()

    return


if __name__ == "__main__":
    """
    diabetes = datasets.load_diabetes(as_frame=True)
    df = diabetes['frame']
    predictors = ['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6']
    response = ['target']
    X = diabetes.data.values
    y = diabetes.target.values
    """
    # iris = datasets.load_iris(as_frame=True)
    # df = iris['frame']
    # predictors = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']
    # response = ['target']
    # variable_type(df,predictors,response)
    # lr(X,y)


from sklearn.ensemble import RandomForestClassifier

# diabetes = datasets.load_diabetes(as_frame=True)
# df = diabetes['frame']
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)
import time

import numpy as np

start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
forest_importances_impurity = pd.Series(importances, index=df.columns[0:5])

fig, ax = plt.subplots()
forest_importances_impurity.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
from sklearn.inspection import permutation_importance

start_time = time.time()
result = permutation_importance(
    forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances_permutation = pd.Series(
    result.importances_mean, index=df.columns[0:5]
)
fig, ax = plt.subplots()
forest_importances_permutation.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
print(
    forest_importances_permutation.to_frame()
    .rename(columns={0: "Feature Importance"})
    .sort_values("Feature Importance", ascending=False)
)


def check_response(df, response):
    # print('Resposne Variable:')
    if df[response].dtypes[0] == "O":
        return "CAT_RES"
    else:
        return "CONT_RES"


def check_predictor(df, i, predictors):
    if df[predictors].dtypes[i] == "O":
        return "CAT_PRED"
    else:
        return "CONT_PRED"


def cont_response_cont_predictor(df, i, response, predictors):
    x = df[df.columns[i]]
    # print(x.values)
    y = df[df.columns[-1]]

    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title="Continuous Response by Continuous Predictor",
        xaxis_title=df.columns[i],
        yaxis_title="Response",
    )
    fig.show()
    return


def cat_resp_cont_predictor(df, i, response, predictors, group_labels):
    # print(df[response])
    group_labels = list(np.unique(df[response].values))
    # Group data together
    hist_data = df[predictors].values

    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
    fig_1.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )
    fig_1.show()
    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig_2.add_trace(
            go.Violin(
                x=np.repeat(curr_group, n),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_2.show()

    return


result_response = check_response(df, response)

print(result_response)
for i in range(0, len(df.columns) - 1):
    # result_response = check_response(df,i,response)
    result_predictor = check_predictor(df, i, predictors)
    if result_response == "CAT_RES":
        if result_predictor == "CAT_PRED":
            print("cat_res_cat_pred()")
        else:
            cat_resp_cont_predictor(df, i, response, predictors, group_labels)
    else:
        if result_predictor == "CAT_PRED":
            print("cont_res_cat_pred()")
        else:
            cont_response_cont_predictor(df, i, response, predictors)
