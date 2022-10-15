# Continuous or Boolean
# Loop through each predictor
# Determine if the predictor is cat/cont
# Automatically generate the necessary plot(s) to inspect it
# Calculate the different ranking algos
# p-value & t-score (continuous predictors only) along with it's plot
# cont_response_cont_predictor - Diabetes
# cat_response_cont_predictor - Iris
# cat_response_cat_predictor - Heart
# cont_response_cat_predictor -

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix


def check_response(df, response):
    if df[response[0]].nunique() == 2:
        if df[response].dtypes[0] == "int64":
            df[response] = df[response].astype("object")
            print(df[response].columns[0], "- Categorical Response")
            return "CAT_RES"
        else:
            print(df[response].columns[0], "- Categorical Response")
            return "CAT_RES"
    elif df[response[0]].nunique() > 2:
        if df[response].dtypes[0] == "O":
            print(df[response].columns[0], "- Categorical Response")
            return "CAT_RES"
        else:
            print(df[response].columns[0], "- Continuous Response")
            return "CONT_RES"


"""
def check_predictor(df,i,predictors):
    if df[predictors].dtypes[i] == "O" or len(df[response[0]].unique()) ==2:
        return('CAT_PRED')
    else:
        return('CONT_PRED')

"""


def check_predictor(df, i, predictors):
    if df[predictors[i]].nunique() == 2:
        if df[predictors[i]].dtypes == "int64":
            print(df[predictors].columns[i], "- Categorical Response saved as numeric")
            df[predictors[i]] = df[predictors[i]].astype("object")
            print(df[predictors].columns[i], "- Categorical Response")
            return "CAT_PRED"
        else:
            print(df[predictors].columns[i], "- Categorical Response")
            return "CAT_PRED"
    elif df[predictors[i]].nunique() > 2:
        if df[predictors[i]].dtypes == "O":
            print(df[predictors].columns[i], "- Categorical Response")
            return "CAT_PRED"
        else:
            # print(df[predictors].columns[i],'- Continuous Response')
            return "CONT_PRED"


def cont_response_cont_predictor(df, i, response, predictors, table):
    print("Called: cont_response_cont_predictor")
    x = df[df.columns[i]]
    # print(x.values)
    y = df[df.columns[-1]]

    predictor = statsmodels.api.add_constant(df.iloc[:, i].T.values)
    linear_regression_model = statsmodels.api.OLS(y, predictor)
    linear_regression_model_fitted = linear_regression_model.fit()
    print(f"Variable: {df.columns[i]}")
    print(linear_regression_model_fitted.summary())
    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title="Continuous Response by Continuous Predictor",
        xaxis_title=df.columns[i],
        yaxis_title="Response",
    )
    fig.show()
    table.append(
        [df.columns[i], "Continuous Predictor", "Continuous Response", t_value, p_value]
    )
    # print(table)
    return


def cat_response_cont_predictor(df, i, response, predictors, table):
    print("Called: cat_response_cont_predictor")
    # print(df[response])
    group_labels = df.iloc[:, -1].unique()
    # Group data together
    hist_data = []
    for j in range(0, len(group_labels)):
        hist_data.append(df[df[response[0]] == group_labels[j]].iloc[:, i].values)

    n = len(df)
    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
    fig_1.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title=df.columns[i],
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
        yaxis_title=df.columns[i],
    )
    fig_2.show()

    x = df.iloc[:, i]
    y = df.iloc[:, -1].astype("category").cat.codes
    y = y.astype("int64")

    predictor = statsmodels.api.add_constant(df.iloc[:, i].T.values)
    log_regression_model = statsmodels.api.Logit(y, predictor)
    log_regression_model_fitted = log_regression_model.fit()
    print(f"Variable: {df.columns[i]}")
    print(log_regression_model_fitted.summary())
    # Get the stats
    t_value = round(log_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(log_regression_model_fitted.pvalues[1])

    table.append(
        [
            df.columns[i],
            "Continuous Predictor",
            "Categorical Response",
            t_value,
            p_value,
        ]
    )

    return


def cat_response_cat_predictor(df, i, response, predictors, table):
    print("Called: cat_response_cat_predictor")
    n = len(df)
    df[response] = df[response].astype("int64")
    x = df.iloc[:, i]
    y = df.iloc[:, -1]

    x_2 = [1 if abs(x_) > 0.5 else 0 for x_ in x]
    y_2 = [1 if abs(y_) > 0.5 else 0 for y_ in y]

    conf_matrix = confusion_matrix(x_2, y_2)

    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    fig_no_relationship.update_layout(
        title="Categorical Predictor by Categorical Response (without relationship)",
        xaxis_title="Response",
        yaxis_title=df.columns[i],
    )
    fig_no_relationship.show()

    x_2 = [1 if abs(x_) > 1.5 else 0 for x_ in x]
    y_2 = [1 if abs(y_) > 1.5 else 0 for y_ in y]

    conf_matrix = confusion_matrix(x_2, y_2)

    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    fig_no_relationship.update_layout(
        title="Categorical Predictor by Categorical Response (with relationship)",
        xaxis_title="Response",
        yaxis_title=df.columns[i],
    )
    fig_no_relationship.show()

    predictor = statsmodels.api.add_constant(df.iloc[:, i].T.values)
    log_regression_model = statsmodels.api.Logit(y, predictor)
    log_regression_model_fitted = log_regression_model.fit()
    print(f"Variable: {df.columns[i]}")
    print(log_regression_model_fitted.summary())
    # Get the stats
    t_value = round(log_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(log_regression_model_fitted.pvalues[1])

    table.append(
        [
            df.columns[i],
            "Categorical Predictor",
            "Categorical Response",
            t_value,
            p_value,
        ]
    )
    return


def cont_response_cat_pred(df, i, response, predictors, table):
    print("Called: cont_response_cat_pred")
    n = len(df)
    fig = px.violin(
        df, y=df.columns[-1], x=df.columns[i], box=True, hover_data=df.columns
    )
    fig.show()
    table.append(
        [df.columns[i], "Categorical Predictor", "Continuous Response", np.nan, np.nan]
    )
    return


def random_forest():
    import time

    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    # diabetes = datasets.load_diabetes(as_frame=True)
    # df = diabetes['frame']
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    forest = RandomForestRegressor(random_state=0)
    forest.fit(X_train, y_train)

    start_time = time.time()
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    forest_importances_impurity = pd.Series(importances, index=df.iloc[:, :-1].columns)

    fig, ax = plt.subplots()
    forest_importances_impurity.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    start_time = time.time()
    result = permutation_importance(
        forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances_permutation = pd.Series(
        result.importances_mean, index=df.iloc[:, :-1].columns
    )
    fig, ax = plt.subplots()
    forest_importances_permutation.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()
    # print(result.importances_mean)
    return result.importances_mean
    # print(forest_importances_permutation.to_frame().rename(columns = {0:'Feature Importance'}).sort_values('Feature Importance',ascending=False))


if __name__ == "__main__":

    """
    #Diabetes
    diabetes = datasets.load_diabetes(as_frame=True)
    df = diabetes['frame']
    predictors = ['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6']
    response = ['target']


    #Salary
    df = pd.read_csv('salary.csv')
    df.head()
    predictors = ['Age','Sex','Location','Experience']
    response = ['Salary']
    """

    # Heart
    df = pd.read_csv("heart.csv")
    df.head()
    predictors = [
        "Age",
        "Sex",
        "ChestPainType",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "RestingECG",
        "MaxHR",
        "ExerciseAngina",
        "Oldpeak",
        "ST_Slope",
    ]
    response = ["HeartDisease"]
    # df[response] = df[response].astype('object')

    result_response = check_response(df, response)
    table = []

    print(result_response)
    for i in range(0, len(df.columns) - 1):
        # result_response = check_response(df,i,response)
        result_predictor = check_predictor(df, i, predictors)
        if result_response == "CAT_RES":
            df[response] = df[response].astype(str)
            if result_predictor == "CAT_PRED":
                df.iloc[:, i] = df.iloc[:, i].astype("category").cat.codes
                cat_response_cat_predictor(df, i, response, predictors, table)
            else:
                cat_response_cont_predictor(df, i, response, predictors, table)
        else:
            if result_predictor == "CAT_PRED":
                df.iloc[:, i] = df.iloc[:, i].astype("category").cat.codes
                cont_response_cat_pred(df, i, response, predictors, table)
            else:
                cont_response_cont_predictor(df, i, response, predictors, table)
    # print(table)
    col = ["Predictor Name", "Predictor Type", "Response Type", "T score", "P Score"]
    table_df = pd.DataFrame(table, columns=col)
    importance_table = random_forest()
    table_df["Importance"] = importance_table
    final_table = table_df.sort_values(by="Importance", axis=0, ascending=False)
    print(final_table)

    # variable_type(df,predictors,response)
    # lr(df,predictors,response)
