import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api
import statsmodels.api as sm
from cat_correlation import cat_cont_correlation_ratio, cat_correlation
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from pyspark.sql import SparkSession
from sklearn import metrics, tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix


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


def split_predictors(X, predictors):
    column_name = []
    column_type = []
    categorical = []
    continuous = []

    for col in predictors:
        if X[col].nunique() == 2:
            if X[col].dtypes == "int64":
                X[col] = X[col].astype("object")
                column_name.append(col)
                column_type.append("Categorical")
                categorical.append(col)
            else:
                column_name.append(col)
                column_type.append("Categorical")
                categorical.append(col)
        elif X[col].nunique() > 2:
            if X[col].dtypes == "O":
                column_name.append(col)
                column_type.append("Categorical")
                categorical.append(col)
            else:
                column_name.append(col)
                column_type.append("Continuous")
                continuous.append(col)

    # print(column_name)
    # print(column_type)
    # print(categorical)
    # print(continuous)

    return (column_name, column_type, categorical, continuous)


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

    # print(hist_data)
    n = len(df)
    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
    fig_1.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title=df.columns[i],
        yaxis_title="Distribution",
    )
    # fig_1.show()

    # urls_d = []
    if not os.path.isdir("Dist-plots"):
        os.mkdir("Dist-plots")
    file_path_dist = f"Dist-plots/{df.columns[i]}-{df.columns[-1]}-plot.html"
    # urls_d.append(file_path)
    fig_1.write_html(file=file_path_dist, include_plotlyjs="cdn")

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

    urls_v = []
    if not os.path.isdir("Violin-plots"):
        os.mkdir("Violin-plots")
    file_path_violin = f"Violin-plots/{df.columns[i]}-{df.columns[-1]}-plot.html"
    urls_v.append(file_path_violin)
    fig_2.write_html(file=file_path_violin, include_plotlyjs="cdn")

    # fig_2.show()

    x = df.iloc[:, i]
    y = df.iloc[:, -1].astype("category").cat.codes
    y = y.astype("int64")
    print(x)
    predictor = statsmodels.api.add_constant(df.iloc[:, i].T.values)
    log_regression_model = statsmodels.api.Logit(y, predictor)
    log_regression_model_fitted = log_regression_model.fit()
    # print(f"Variable: {df.columns[i]}")
    # print(log_regression_model_fitted.summary())
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
            file_path_dist,
            file_path_violin,
        ]
    )

    return


def cat_response_cat_predictor(df, i, response, predictors, table):
    print("Called: cat_response_cat_predictor")
    n = len(df)
    print(n)
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
    print(n)
    fig = px.violin(
        df, y=df.columns[-1], x=df.columns[i], box=True, hover_data=df.columns
    )
    fig.show()
    table.append(
        [df.columns[i], "Categorical Predictor", "Continuous Response", np.nan, np.nan]
    )
    return


# Create a spark session
spark = (
    SparkSession.builder.config(
        "spark.jars",
        "/Users/zack/Documents/SDSU/Fall 2022/mysql-connector-java-5.1.46/mysql-connector-java-5.1.46.jar",
    )
    .master("local")
    .appName("HW5")
    .getOrCreate()
)


# Create a function to load the data from MariaDB
def load_data(query):
    table_data = (
        spark.read.format("jdbc")
        .option("url", "jdbc:mysql://localhost:3306/baseball")
        .option("driver", "com.mysql.jdbc.Driver")
        .option("query", query)
        .option("user", "root")
        .option("password", "believe")
        .load()
    )
    return table_data


def convert_to_pandas_df(spark_df):
    data = spark_df.toPandas()
    return data


def transform_data(data):

    data["local_date"] = pd.to_datetime(data["local_date"])
    data["Home_Team_PlateApperance"] = pd.to_numeric(data["Home_Team_PlateApperance"])
    data["Away_Team_PlateApperance"] = pd.to_numeric(data["Away_Team_PlateApperance"])
    data["Home_Team_Slugging_Percentage"] = pd.to_numeric(
        data["Home_Team_Slugging_Percentage"]
    )
    data["Away_Team_Slugging_Percentage"] = pd.to_numeric(
        data["Away_Team_Slugging_Percentage"]
    )
    data["Home_Team_Batting_Average"] = pd.to_numeric(data["Home_Team_Batting_Average"])
    data["Away_Team_Batting_Average"] = pd.to_numeric(data["Away_Team_Batting_Average"])

    new_data = data.loc[:, data.columns != "local_date"]
    # print(data.dtypes)
    # print(data.columns)
    train = new_data.loc[data["local_date"] < "2010-12-12"]
    test = new_data.loc[data["local_date"] >= "2010-12-12"]
    X_train = train.iloc[:, :-1]
    X_test = test.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    y_test = test.iloc[:, -1]

    return X_train, X_test, y_train, y_test, new_data


def correlation_metrics_cont_cont(X, predictors, continuous):
    corr = X[continuous].corr(method="pearson")

    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=corr.columns.values.tolist(),
        y=corr.index.values.tolist(),
        zmin=-1,
        zmax=1,
        colorscale="viridis",
        showscale=True,
        hoverongaps=True,
    )
    fig.update_layout(
        title="Continuous Predictor by Continuous Predictor",
        xaxis_title="Continuous Predictors",
        yaxis_title="Continuous Predictors",
    )
    # print(corr)
    temp = (
        corr.stack()
        .rename_axis(("Predictor 1", "Predictor 2"))
        .reset_index(name="Correlation Value")
    )
    # print(temp)
    mask_dups = (
        temp[["Predictor 1", "Predictor 2"]].apply(frozenset, axis=1).duplicated()
    ) | (temp["Predictor 1"] == temp["Predictor 2"])
    temp = temp[~mask_dups]
    # print(temp)
    fig.show()
    urls = []
    for i in range(0, len(continuous)):
        for j in range(0, len(continuous)):
            if i != j and i < j:
                fig_plots = px.scatter(
                    X, x=continuous[i], y=continuous[j], trendline="ols"
                )
                if not os.path.isdir("Cont-Cont-plots"):
                    os.mkdir("Cont-Cont-plots")
                file_path = f"Cont-Cont-plots/{continuous[i]}-{continuous[j]}-plot.html"
                urls.append(file_path)
                fig_plots.write_html(file=file_path, include_plotlyjs="cdn")
    temp["URL"] = urls
    # table = temp.style.set_properties(**{'border': '1.3px solid black'})
    temp = temp.sort_values("Correlation Value", ascending=False)
    temp = temp.reset_index(drop=True)
    table = temp.style.set_properties(**{"border": "1.3px solid black"}).format(
        {"URL": make_clickable}, escape="html"
    )
    cont_cont_table = table.to_html("Continuous-Continuous.html")
    print(cont_cont_table)

    # fig_plots.show()


def correlation_metrics_cat_cat(X, predictors, categorical):
    # Try the approach by Julian
    Predictor1 = []
    Predictor2 = []
    Corr_Ratio = []
    for i in range(0, len(categorical)):
        for j in range(0, len(categorical)):
            Predictor1.append(categorical[i])
            Predictor2.append(categorical[j])
            Corr_Ratio.append(cat_correlation(X[categorical[i]], X[categorical[j]]))

    cat_cat_df = pd.DataFrame(
        list(zip(Predictor1, Predictor2, Corr_Ratio)),
        columns=["Predictor 1", "Predictor 2", "Correlation Value"],
    )
    # print(cat_cat_df)

    cat_cat_df_corr = cat_cat_df.pivot(
        index="Predictor 1", columns="Predictor 2", values="Correlation Value"
    )

    fig = ff.create_annotated_heatmap(
        z=cat_cat_df_corr.values,
        x=cat_cat_df_corr.columns.values.tolist(),
        y=cat_cat_df_corr.index.values.tolist(),
        zmin=-1,
        zmax=1,
        colorscale="viridis",
        showscale=True,
        hoverongaps=True,
        font_colors=["black"],
    )

    fig.update_layout(
        title="Categorical Predictor by Categorical Predictor",
        xaxis_title="Categorical Predictors",
        yaxis_title="Categorical Predictors",
    )
    fig.show()

    urls = []
    for i in range(0, len(categorical)):
        for j in range(0, len(categorical)):
            if i != j and i < j:
                cat1_unqiue = X[categorical[i]].unique().tolist()
                cat2_unique = X[categorical[j]].unique().tolist()
                grouped_data = X.groupby([categorical[i], categorical[j]])
                # print(grouped_data.groups)
                heatmap_layout = []
                for val2 in cat2_unique:
                    row_data = []
                    for val1 in cat1_unqiue:
                        if (val1, val2) not in grouped_data.groups:
                            row_data.append(0)
                        else:
                            row_data.append(len(grouped_data.groups.get((val1, val2))))
                    heatmap_layout.append(row_data)
                # print(heatmap_layout)

                fig_plot = ff.create_annotated_heatmap(
                    heatmap_layout,
                    x=cat1_unqiue,
                    y=cat2_unique,
                    colorscale="viridis",
                    showscale=True,
                    hoverongaps=True,
                    font_colors=["black"],
                )
                fig_plot.update_layout(
                    title="Categorical vs Categorical",
                    xaxis_title=categorical[i],
                    yaxis_title=categorical[j],
                )
                if not os.path.isdir("Cat-Cat-plots"):
                    os.mkdir("Cat-Cat-plots")
                file_path = f"Cat-Cat-plots/{categorical[i]}-{categorical[j]}-plot.html"
                urls.append(file_path)
                fig_plot.write_html(file=file_path, include_plotlyjs="cdn")
                # fig_plot.show()
    mask_dups = (
        cat_cat_df[["Predictor 1", "Predictor 2"]].apply(frozenset, axis=1).duplicated()
    ) | (cat_cat_df["Predictor 1"] == cat_cat_df["Predictor 2"])
    cat_cat_df = cat_cat_df[~mask_dups]
    cat_cat_df["URL"] = urls
    cat_cat_df = cat_cat_df.sort_values("Correlation Value", ascending=False)
    cat_cat_df = cat_cat_df.reset_index(drop=True)
    table = cat_cat_df.style.set_properties(**{"border": "1.3px solid black"}).format(
        {"URL": make_clickable}, escape="html"
    )
    cat_cat_table = table.to_html("Categorical-Categorical.html")
    print(cat_cat_table)


def correlation_metrics_cat_cont(X, predictors, continuous, categorical):
    # Try the approach by Julian
    Predictor1 = []
    Predictor2 = []
    Corr_Ratio = []
    for i in range(0, len(categorical)):
        for j in range(0, len(continuous)):
            Predictor1.append(categorical[i])
            Predictor2.append(continuous[j])
            Corr_Ratio.append(
                cat_cont_correlation_ratio(X[categorical[i]], X[continuous[j]])
            )

    cat_cont_df = pd.DataFrame(
        list(zip(Predictor1, Predictor2, Corr_Ratio)),
        columns=["Predictor 1", "Predictor 2", "Correlation Value"],
    )
    # print(cat_cont_df)

    cat_cont_df_corr = cat_cont_df.pivot(
        index="Predictor 1", columns="Predictor 2", values="Correlation Value"
    )
    # print(cat_cont_df_corr)

    fig = ff.create_annotated_heatmap(
        z=cat_cont_df_corr.values,
        x=cat_cont_df_corr.columns.values.tolist(),
        y=cat_cont_df_corr.index.values.tolist(),
        zmin=-1,
        zmax=1,
        colorscale="viridis",
        showscale=True,
        hoverongaps=True,
        font_colors=["black"],
    )

    fig.update_layout(
        title="Categorical Predictor by Continuous Predictor",
        xaxis_title="Continuous Predictors",
        yaxis_title="Categorical Predictors",
    )
    fig.show()
    urls = []
    for i in range(len(categorical)):
        for j in range(len(continuous)):
            cont_values = []
            group_labels = X[categorical[i]].unique().tolist()
            for val1 in group_labels:
                abc = X.loc[X[categorical[i]] == val1]
                cont_values.append(abc[continuous[j]].to_list())
            fig_plots = ff.create_distplot(
                cont_values,
                group_labels,
                bin_size=((X[continuous[j]].max() - X[continuous[j]].min()) / 20),
            )
            fig_plots.update_layout(
                title="Categorical vs Continuous",
                xaxis_title=continuous[j],
                legend_title=categorical[i],
                yaxis_title="Distribution",
            )
            if not os.path.isdir("Cat-Cont-plots"):
                os.mkdir("Cat-Cont-plots")
            file_path = f"Cat-Cont-plots/{categorical[i]}-{continuous[j]}-plot.html"
            urls.append(file_path)
            fig_plots.write_html(file=file_path, include_plotlyjs="cdn")
            # fig_plots.show()

    cat_cont_df["URL"] = urls
    cat_cont_df = cat_cont_df.sort_values("Correlation Value", ascending=False)
    cat_cont_df = cat_cont_df.reset_index(drop=True)
    table = cat_cont_df.style.set_properties(**{"border": "1.3px solid black"}).format(
        {"URL": make_clickable}, escape="html"
    )
    cat_cont_table = table.to_html("Categorical-Continuous.html")
    print(cat_cont_table)

    return


def build_test_models(X_train, X_test, y_train, y_test):

    # DECISION TREE CLASSIFIER

    DT_clf = tree.DecisionTreeClassifier(random_state=0, max_depth=8)
    DT_Fitted = DT_clf.fit(X_train, y_train)
    # print(clf.feature_importances_)
    y_pred_DT = DT_Fitted.predict(X_test)
    accuracy_DT = metrics.accuracy_score(y_test, y_pred_DT)
    F1_Macro_DT = metrics.f1_score(y_test, y_pred_DT, average="macro")
    F1_Micro_DT = metrics.f1_score(y_test, y_pred_DT, average="micro")
    F1_Weighted_DT = metrics.f1_score(y_test, y_pred_DT, average="weighted")

    print("Decision Tree Accuracy:", accuracy_DT)
    print("F1 Score - Macro", F1_Macro_DT)
    print("F1 Score - Micro", F1_Micro_DT)
    print("F1 Score - Weighted", F1_Weighted_DT)

    Precision_Macro_DT = metrics.precision_score(y_test, y_pred_DT, average="macro")
    Precision_Micro_DT = metrics.precision_score(y_test, y_pred_DT, average="micro")
    Precision_Weighted_DT = metrics.precision_score(
        y_test, y_pred_DT, average="weighted"
    )
    print("Precision Score - Macro", Precision_Macro_DT)
    print("Precision Score - Micro", Precision_Micro_DT)
    print("Precision Score - Weighted", Precision_Weighted_DT)

    # LOGISTIC REGRESSION

    log_reg = sm.Logit(y_train, X_train).fit()
    # print(log_reg.summary())
    y_pred_LOGIT = log_reg.predict(X_test)

    accuracy_LOGIT = metrics.accuracy_score(y_test, round(y_pred_LOGIT))
    F1_Macro_LOGIT = metrics.f1_score(y_test, round(y_pred_LOGIT), average="macro")
    F1_Micro_LOGIT = metrics.f1_score(y_test, round(y_pred_LOGIT), average="micro")
    F1_Weighted_LOGIT = metrics.f1_score(
        y_test, round(y_pred_LOGIT), average="weighted"
    )

    print("Logistic Regression StatsModel Accuracy = ", accuracy_LOGIT)
    print("F1 Score - Macro", F1_Macro_LOGIT)
    print("F1 Score - Micro", F1_Micro_LOGIT)
    print("F1 Score - Weighted", F1_Weighted_LOGIT)

    Precision_Macro_LOGIT = metrics.precision_score(
        y_test, round(y_pred_LOGIT), average="macro"
    )
    Precision_Micro_LOGIT = metrics.precision_score(
        y_test, round(y_pred_LOGIT), average="micro"
    )
    Precision_Weighted_LOGIT = metrics.precision_score(
        y_test, round(y_pred_LOGIT), average="weighted"
    )

    print("Precision Score - Macro", Precision_Macro_LOGIT)
    print("Precision Score - Micro", Precision_Micro_LOGIT)
    print("Precision Score - Weighted", Precision_Weighted_LOGIT)

    # RANDOM FOREST CLASSIFIER

    RF_clf = RandomForestClassifier(max_depth=20, random_state=0)
    RF_Fitted = RF_clf.fit(X_train, y_train)
    y_pred_RF = RF_Fitted.predict(X_test)

    accuracy_RF = metrics.accuracy_score(y_test, y_pred_RF)
    F1_Macro_RF = metrics.f1_score(y_test, y_pred_RF, average="macro")
    F1_Micro_RF = metrics.f1_score(y_test, y_pred_RF, average="micro")
    F1_Weighted_RF = metrics.f1_score(y_test, y_pred_RF, average="weighted")

    print("Random Forest Accuracy:", accuracy_RF)
    print("F1 Score - Macro", F1_Macro_RF)
    print("F1 Score - Micro", F1_Micro_RF)
    print("F1 Score - Weighted", F1_Weighted_RF)

    Precision_Macro_RF = metrics.precision_score(y_test, y_pred_RF, average="macro")
    Precision_Micro_RF = metrics.precision_score(y_test, y_pred_RF, average="micro")
    Precision_Weighted_RF = metrics.precision_score(
        y_test, y_pred_RF, average="weighted"
    )

    print("Precision Score - Macro", Precision_Macro_RF)
    print("Precision Score - Micro", Precision_Micro_RF)
    print("Precision Score - Weighted", Precision_Weighted_RF)

    feature_importance = RF_clf.feature_importances_

    # Gradient Boosting Classifier

    GB_clf = GradientBoostingClassifier(
        learning_rate=0.15, max_features=2, max_depth=12, random_state=0
    )
    GB_Fitted = GB_clf.fit(X_train, y_train)
    y_pred_GB = GB_Fitted.predict(X_test)
    accuracy_GB = metrics.accuracy_score(y_test, y_pred_GB)
    F1_Macro_GB = metrics.f1_score(y_test, y_pred_GB, average="macro")
    F1_Micro_GB = metrics.f1_score(y_test, y_pred_GB, average="micro")
    F1_Weighted_GB = metrics.f1_score(y_test, y_pred_GB, average="weighted")

    print("Gradient Boosting Accuracy:", accuracy_GB)
    print("F1 Score - Macro", F1_Macro_GB)
    print("F1 Score - Micro", F1_Micro_GB)
    print("F1 Score - Weighted", F1_Weighted_GB)

    Precision_Macro_GB = metrics.precision_score(y_test, y_pred_GB, average="macro")
    Precision_Micro_GB = metrics.precision_score(y_test, y_pred_GB, average="micro")
    Precision_Weighted_GB = metrics.precision_score(
        y_test, y_pred_GB, average="weighted"
    )

    print("Precision Score - Macro", Precision_Macro_GB)
    print("Precision Score - Micro", Precision_Micro_GB)
    print("Precision Score - Weighted", Precision_Weighted_GB)

    return feature_importance


def mean_of_response_cont_cont(X, continuous, response):
    Predictor1 = []
    Predictor2 = []
    Unweighted_Mean_of_Response = []
    Weighted_Mean_of_Response = []
    urls = []
    bin_size = 54
    bin_size_2d = bin_size * bin_size
    for i in range(0, len(continuous)):
        for j in range(0, len(continuous)):
            if i != j and i < j:
                Predictor1.append(continuous[i])
                Predictor2.append(continuous[j])

                population_mean = X[response].mean()
                population_count = X[response].count()
                bins = {}
                bins["cont1_bins"] = pd.cut(
                    X[continuous[i]], bin_size, include_lowest=True, duplicates="drop"
                )
                bins["cont2_bins"] = pd.cut(
                    X[continuous[j]], bin_size, include_lowest=True, duplicates="drop"
                )
                bins_df = pd.DataFrame(bins)
                bin_columns = bins_df.columns.to_list()
                # print(bins_df)
                filtered_df = X.filter([continuous[i], continuous[j], response], axis=1)
                joined_bin = filtered_df.join(bins_df)
                # print(joined_bin.columns)

                grouped_bin = joined_bin.groupby(bin_columns)
                bin_mean = grouped_bin.mean().unstack()
                # print(bin_mean.columns)

                counts = grouped_bin.count().unstack()
                response_count = counts[response]
                res_means_df = bin_mean[response]
                res_means_diff_df = res_means_df - population_mean
                res_weights_df = response_count / population_count

                res_means_diff_weighted_df = res_means_diff_df * res_weights_df
                plot_x = res_means_df.columns.map(lambda x: (x.left + x.right) / 2)
                plot_y = res_means_df.index.map(lambda x: (x.left + x.right) / 2)
                plot_z = res_means_diff_df.values

                diff_mean_unweighted = (
                    res_means_diff_df.pow(2).sum().sum() / bin_size_2d
                )
                diff_mean_weighted = (
                    res_means_diff_weighted_df.pow(2).sum().sum() / bin_size_2d
                )
                Unweighted_Mean_of_Response.append(diff_mean_unweighted)
                Weighted_Mean_of_Response.append(diff_mean_weighted)

                # print(a, b,diff_mean_unweighted,diff_mean_weighted)
                fig_unweighted = go.Figure(
                    data=go.Heatmap(
                        z=plot_z,
                        x=plot_x,
                        y=plot_y,
                        hoverongaps=True,
                        colorscale="Viridis",
                    )
                )
                fig_unweighted.update_layout(
                    title=f"{continuous[i]} VS {continuous[j]} MOR",
                    xaxis_title=continuous[i],
                    yaxis_title=continuous[j],
                )
                # fig_unweighted.show()
                if not os.path.isdir("Brute-Force-plots"):
                    os.mkdir("Brute-Force-plots")
                file_path = (
                    f"Brute-Force-plots/{continuous[i]}-{continuous[j]}-plot.html"
                )
                urls.append(file_path)
                fig_unweighted.write_html(file=file_path, include_plotlyjs="cdn")
    brute_cont_cont_df = pd.DataFrame(
        list(
            zip(
                Predictor1,
                Predictor2,
                Unweighted_Mean_of_Response,
                Weighted_Mean_of_Response,
                urls,
            )
        ),
        columns=[
            "Predictor 1",
            "Predictor 2",
            "Unweighted Mean of Response",
            "Weighted Mean of Response",
            "URL",
        ],
    )
    brute_cont_cont_df = brute_cont_cont_df.sort_values(
        "Weighted Mean of Response", ascending=False
    )
    brute_cont_cont_df = brute_cont_cont_df.reset_index(drop=True)
    table = brute_cont_cont_df.style.set_properties(
        **{"border": "1.3px solid black"}
    ).format({"URL": make_clickable}, escape="html")
    table.to_html("Brute_Force_Continuous_Continuous.html")


def mean_of_response_cat_cat(X, categorical, response):
    Predictor1 = []
    Predictor2 = []
    Unweighted_Mean_of_Response = []
    Weighted_Mean_of_Response = []
    urls = []
    for a, b in combinations(categorical, r=2):
        n_2d_bins = len(X[a].unique()) * len(X[b].unique())
        Predictor1.append(a)
        Predictor2.append(b)
        res_mean = X[response].mean()
        res_count = X[response].count()

        grouped_bin = X.filter([a, b, response], axis=1).groupby([a, b])
        bin_mean = grouped_bin.mean().unstack()
        counts = grouped_bin.count().unstack()
        response_mean = bin_mean[response]
        res_means_diff = response_mean.sub(res_mean)
        response_count = counts[response]

        res_weights_df = response_count.div(res_count)
        res_means_diff_weighted_df = res_means_diff.mul(res_weights_df)
        diff_mean_unweighted = (res_means_diff.pow(2).sum().sum() / n_2d_bins) ** 0.5
        diff_mean_weighted = (
            res_means_diff_weighted_df.pow(2).sum().sum() / n_2d_bins
        ) ** 0.5

        Unweighted_Mean_of_Response.append(diff_mean_unweighted)
        Weighted_Mean_of_Response.append(diff_mean_weighted)

        plot_x = response_mean.columns
        plot_y = response_mean.index
        plot_z = res_means_diff.values

        fig_unweighted = go.Figure(
            data=go.Heatmap(
                z=plot_z,
                x=plot_x,
                y=plot_y,
                colorbar={"title": response},
                colorscale="viridis",
                hoverongaps=True,
            )
        )
        fig_unweighted.update_layout(
            title=f"{a} VS {b} MOR",
            xaxis_title=a,
            yaxis_title=b,
        )
        # fig_unweighed.show()
        if not os.path.isdir("Brute-Force-plots"):
            os.mkdir("Brute-Force-plots")
        file_path = f"Brute-Force-plots/{a}-{b}-plot.html"
        urls.append(file_path)
        fig_unweighted.write_html(file=file_path, include_plotlyjs="cdn")

    brute_cat_cat_df = pd.DataFrame(
        list(
            zip(
                Predictor1,
                Predictor2,
                Unweighted_Mean_of_Response,
                Weighted_Mean_of_Response,
                urls,
            )
        ),
        columns=[
            "Predictor 1",
            "Predictor 2",
            "Unweighted Mean of Response",
            "Weighted Mean of Response",
            "URL",
        ],
    )
    brute_cat_cat_df = brute_cat_cat_df.sort_values(
        "Weighted Mean of Response", ascending=False
    )
    brute_cat_cat_df = brute_cat_cat_df.reset_index(drop=True)
    table = brute_cat_cat_df.style.set_properties(
        **{"border": "1.3px solid black"}
    ).format({"URL": make_clickable}, escape="html")
    brute_cont_cont_table = table.to_html("Brute_Force_Categorical_Categorical.html")
    print(brute_cont_cont_table)
    return


def make_clickable(val):
    return '<a href="{}">{}</a>'.format(val, val)


def mean_of_response(df, predictors, response):
    # print(df.dtypes)
    for predictor in predictors:
        bin = pd.cut(df[predictor], 10)
        # w = bin.value_counts() / sum(bin.value_counts())  # Population Proportion
        # w = w.reset_index()
        pop_mean = np.mean(df[response])  # population mean
        response_mean = df[response].groupby(bin).apply(np.mean).reset_index()
        print(response_mean, pop_mean)
        # unweigh_diff = np.sum(np.square(response_mean.iloc[:, 1] - pop_mean)) / len( response_mean)
        # unweighted difference with mean
        # weigh_diff = np.sum(w.iloc[:, 1] * np.square(response_mean.iloc[:, 1] - pop_mean)) / len(response_mean)
        # weighted difference with mean
        # print(f"unweighted difference with mean for {predictor}: {unweigh_diff}")
        # print(f"unweighted difference with mean for {predictor}: {weigh_diff}")
    return


def main():
    # Loading the Features Table
    features_sql = """ SELECT * FROM FEATURES """
    Features = load_data(features_sql)
    result_response = "CAT_RES"
    # Converting Spark Dataframe to Pandas Dataframe
    df = convert_to_pandas_df(Features)

    # Split Data into Train and test by Date
    X_train, X_test, y_train, y_test, new_data = transform_data(df)
    df = new_data
    predictors = [
        "game_id",
        "Home_Team_ID",
        "Away_Team_ID",
        "Home_Team_Strikouts",
        "Away_Team_Strikouts",
        "Home_Team_PlateApperance",
        "Away_Team_PlateApperance",
        "Home_Team_Single",
        "Away_Team_Single",
        "Home_Team_Double",
        "Away_Team_Double",
        "Home_Team_Triple",
        "Away_Team_Triple",
        "Home_Team_Slugging_Percentage",
        "Away_Team_Slugging_Percentage",
        "Home_Team_Batting_Average",
        "Away_Team_Batting_Average",
        "Home_Team_Walk_strikeout_ratio",
        "Away_Team_Walk_strikeout_ratio",
        "Home_Team_Ground_fly_ball_ratio",
        "Away_Team_Ground_fly_ball_ratio",
        "Home_Team_Intentional_Walk",
        "Away_Team_Intentional_Walk",
        "Home_Team_At_bats_per_home_run",
        "Away_Team_At_bats_per_home_run",
        "Home_Team_Home_runs_per_hit",
        "Away_Team_Home_runs_per_hit",
    ]
    response = ["HomeTeamWins"]
    table = []

    # Split Data into Categorical and Continuous Variables
    column_name, column_type, categorical, continuous = split_predictors(df, predictors)

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

    # Build Models using this data
    importance = build_test_models(X_train, X_test, y_train, y_test)
    # load_dataset()
    # mean_of_response(df,predictors,response)
    # print(table)
    col = [
        "Predictor_Name",
        "Predictor_Type",
        "Response_Type",
        "T_Score",
        "P_Score",
        "Distribution_Plot",
        "Violin_Plot",
    ]
    table_df = pd.DataFrame(table, columns=col)
    table_df["Feature_Importance"] = importance
    table_df = table_df.sort_values(by="Feature_Importance", axis=0, ascending=False)

    final_table = table_df.style.set_properties(
        **{"border": "1.3px solid black"}
    ).format(
        {"Distribution_Plot": make_clickable, "Violin_Plot": make_clickable},
        escape="html",
    )
    final_table.to_html("HW4_Plots.html")

    # Compute the correlation metrics for predictors
    X = df[predictors]
    # y = df[response[0]]
    correlation_metrics_cont_cont(X, predictors, continuous)
    # correlation_metrics_cat_cat(X, predictors, categorical)
    # correlation_metrics_cat_cont(X, predictors, continuous, categorical)
    print(importance)
    print(type(importance))

    # Brute Force Mean of Response
    response = "HomeTeamWins"
    # print(df.dtypes)
    df["HomeTeamWins"] = df["HomeTeamWins"].astype("int64")
    mean_of_response_cont_cont(df, continuous, response)
    # mean_of_response_cat_cat(df, categorical, response)
    # mean_of_response_cat_cont(data, categorical, continuous,response)


if __name__ == "__main__":
    sys.exit(main())
