import os
import sys
from itertools import combinations

import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import pandas as pd
import seaborn as sns

# import sqlalchemy
import statsmodels.api
import statsmodels.api as sm
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn import metrics, tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.svm import SVC

# from sqlalchemy import text


def MeanOfResponse(df, predictors, response):
    if df[response[0]].dtypes == "O":
        df[response[0]] = df[response[0]].astype("int64")
    # print(response[0])
    print("Inside Mean of Response")
    print(df.dtypes)
    pop_mean = df[response[0]].values.mean()
    # print(pop_mean)
    MOR_Table = pd.DataFrame()
    final_MOR_1D_table = pd.DataFrame(
        columns=[
            "Response",
            "Predictor",
            "Mean_Squared_Error",
            "Mean_Squared_Error_Weighted",
            "URL",
        ]
    )
    for i in range(0, len(predictors)):
        df_copy = df[
            [predictors[i], response[0]]
        ].copy()  # create a copy of predictor, response pair
        df_copy["Predictor_Bins"] = pd.cut(
            df_copy[predictors[i]], 10, include_lowest=False, duplicates="drop"
        )
        df_copy["Lower_Bin"] = (
            df_copy["Predictor_Bins"].apply(lambda x: x.left).astype(float)
        )
        df_copy["Upper_Bin"] = (
            df_copy["Predictor_Bins"].apply(lambda x: x.right).astype(float)
        )

        df_copy["Bin_Centre"] = (df_copy["Lower_Bin"] + df_copy["Upper_Bin"]) / 2
        # print(df_copy.head())
        # print(df_copy.index)
        bin_mean = df_copy.groupby(by=["Lower_Bin", "Upper_Bin"]).mean().reset_index()
        # print(bin_mean.head())

        bin_count = df_copy.groupby(by=["Lower_Bin", "Upper_Bin"]).count().reset_index()
        bin_center = (
            df_copy.groupby(by=["Lower_Bin", "Upper_Bin"]).median().reset_index()
        )
        # print('bin_count')

        # print(df_copy["Bin_Centre"])
        # print(len(df_copy["Bin_Centre"]))
        # print(df_copy["Predictor_Bins"])
        # print(pd.cut(df_copy["Predictor_Bins"], 10, include_lowest=True, duplicates="drop"))
        MOR_Table["Bin_Count"] = bin_count[predictors[i]]
        MOR_Table["Bin_Mean"] = bin_mean[response[0]]
        MOR_Table["Bin_Centre"] = bin_center[predictors[i]]
        MOR_Table["Population_Mean"] = pop_mean
        MOR_Table["Mean_diff"] = round(
            MOR_Table["Bin_Mean"] - MOR_Table["Population_Mean"],
            6,
        )
        MOR_Table["mean_squared_diff"] = round((MOR_Table["Mean_diff"]) ** 2, 6)
        MOR_Table["Weight"] = MOR_Table["Bin_Count"] / df[response[0]].count()
        MOR_Table["mean_squared_diff_weighted"] = (
            MOR_Table["mean_squared_diff"] * MOR_Table["Weight"]
        )
        # MOR_Table = MOR_Table.reset_index()
        MOR_Table["mean_squared_diff"] = MOR_Table["mean_squared_diff"].mean()
        # print(MOR_Table.head())
        mean_squared_diff = round(
            (MOR_Table["mean_squared_diff"].sum()),
            6,
        )
        mean_squared_diff_weighted = round(
            MOR_Table["mean_squared_diff_weighted"].sum(), 6
        )
        x_axis = MOR_Table["Bin_Centre"]  # Check again
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # print(bin_count[predictors[i]])
        fig.add_trace(
            go.Bar(
                x=x_axis,
                y=MOR_Table["Bin_Count"],
                name="Population",
                opacity=0.5,
            ),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(x=x_axis, y=MOR_Table["Bin_Mean"], name="Bin Mean"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=[x_axis.min(), x_axis.max()],
                y=[
                    MOR_Table["Population_Mean"][0],
                    MOR_Table["Population_Mean"][0],
                ],
                mode="lines",
                line=dict(color="green", width=2),
                name="Population Mean",
            )
        )

        fig.add_hline(MOR_Table["Population_Mean"][0], line_color="green")

        title = f" Mean of Response vs Bin ({predictors[i]})"
        # Add figure title
        fig.update_layout(title_text=f"Mean of Response of {predictors[i]}")

        # Set x-axis title
        fig.update_xaxes(title_text=f"Predictor - {predictors[i]} Bin")

        # Set y-axes titles
        fig.update_yaxes(title_text="Population", secondary_y=True)
        fig.update_yaxes(title_text=f"Response - {response[0]}", secondary_y=False)
        # fig.show()

        urls = []

        if not os.path.isdir("Mean_Of_Response_1D_plots"):
            os.mkdir("Mean_Of_Response_1D_plots")
        file_path = f"Mean_Of_Response_1D_plots/{predictors[i]}-{response[0]}-plot.html"
        urls.append(file_path)
        fig.write_html(file=file_path, include_plotlyjs="cdn")
        # Table for Single Predictor
        # print(MOR_Table)
        # print(urls[0])
        # print(response[0],predictors[i],mean_squared_diff,mean_squared_diff_weighted,urls)

        final_MOR_1D_table = final_MOR_1D_table.append(
            {
                "Response": response[0],
                "Predictor": predictors[i],
                "Mean_Squared_Error": mean_squared_diff,
                "Mean_Squared_Error_Weighted": mean_squared_diff_weighted,
                "URL": urls[0],
            },
            ignore_index=True,
        )

    final_MOR_1D_table = final_MOR_1D_table.sort_values(
        "Mean_Squared_Error_Weighted", ascending=False
    )
    # final_MOR_1D_table = final_MOR_1D_table.reset_index(drop=True)
    final_MOR_1D_table = final_MOR_1D_table.style.set_properties(
        **{"border": "1.3px solid black"}
    ).format({"URL": make_clickable}, escape="html")
    final_MOR_1D_table.to_html("Mean_Of_Response_1D.html")


def Brute_Force_cont_cont(df, predictors, continuous, response):
    pop_mean = df[response[0]].values.mean()
    # print(df.head())
    Brute_Force_Table = pd.DataFrame()
    final_Brute_Force_table = pd.DataFrame(
        columns=[
            "Predictor_1",
            "Predictor_2",
            "Mean_Squared_Error",
            "Mean_Squared_Error_Weighted",
            "URL",
        ]
    )
    print(type(final_Brute_Force_table))
    for a, b in combinations(continuous, 2):
        df_copy = df[[a, b, response[0]]].copy()
        # print(df_copy.head())
        df_copy["Bin_1"] = pd.cut(
            df_copy[a], 10, include_lowest=True, duplicates="drop"
        )
        df_copy["Bin_2"] = pd.cut(
            df_copy[b], 10, include_lowest=True, duplicates="drop"
        )
        df_copy["Lower_Bin_1"] = df_copy["Bin_1"].apply(lambda x: x.left).astype(float)
        df_copy["Upper_Bin_1"] = df_copy["Bin_1"].apply(lambda x: x.right).astype(float)
        df_copy["Lower_Bin_2"] = df_copy["Bin_2"].apply(lambda x: x.left).astype(float)
        df_copy["Upper_Bin_2"] = df_copy["Bin_2"].apply(lambda x: x.right).astype(float)
        df_copy["Bin_Centre_1"] = (df_copy["Lower_Bin_1"] + df_copy["Upper_Bin_1"]) / 2
        df_copy["Bin_Centre_2"] = (df_copy["Lower_Bin_2"] + df_copy["Upper_Bin_2"]) / 2

        bin_mean = (
            df_copy.groupby(by=["Bin_Centre_1", "Bin_Centre_2"]).mean().reset_index()
        )
        # print(bin_mean)
        bin_count = (
            df_copy.groupby(by=["Bin_Centre_1", "Bin_Centre_2"]).count().reset_index()
        )
        bin_center_1 = (
            df_copy.groupby(by=["Lower_Bin_1", "Upper_Bin_1"]).median().reset_index()
        )
        bin_center_2 = (
            df_copy.groupby(by=["Lower_Bin_2", "Upper_Bin_2"]).median().reset_index()
        )

        Brute_Force_Table["Bin_Count"] = bin_count[a]
        Brute_Force_Table["Bin_Mean"] = bin_mean[response[0]]

        Brute_Force_Table["Bin_Centre_1"] = bin_center_1[a]
        Brute_Force_Table["Bin_Centre_2"] = bin_center_2[a]

        Brute_Force_Table["Population_Mean"] = pop_mean
        Brute_Force_Table["Mean_diff"] = round(
            Brute_Force_Table["Bin_Mean"] - Brute_Force_Table["Population_Mean"], 6
        )
        Brute_Force_Table["mean_squared_diff"] = round(
            (Brute_Force_Table["Mean_diff"]) ** 2, 6
        )
        Brute_Force_Table["Weight"] = (
            Brute_Force_Table["Bin_Count"] / df[response[0]].count()
        )
        Brute_Force_Table["mean_squared_diff_weighted"] = (
            Brute_Force_Table["mean_squared_diff"] * Brute_Force_Table["Weight"]
        )
        # Brute_Force_Table = Brute_Force_Table.reset_index()
        Brute_Force_Table["mean_squared_diff"] = Brute_Force_Table[
            "mean_squared_diff"
        ].mean()
        mean_squared_diff = round((Brute_Force_Table["mean_squared_diff"].sum()), 6)
        mean_squared_diff_weighted = round(
            Brute_Force_Table["mean_squared_diff_weighted"].sum(), 6
        )

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=Brute_Force_Table["Bin_Centre_1"].values,  # Check AGAIN
                y=Brute_Force_Table["Bin_Centre_2"].values,  # Check AGAIN
                z=Brute_Force_Table["Mean_diff"],
                text=Brute_Force_Table["Mean_diff"],
                texttemplate="%{text}",
            )
        )

        title = f"Brute Force Mean of Response for {a} by {b}"

        fig.update_layout(
            title=title,
            xaxis_title=f"{a}",
            yaxis_title=f"{b}",
        )

        urls = []

        if not os.path.isdir("Brute_Force_plots"):
            os.mkdir("Brute_Force_plots")
        file_path = f"Brute_Force_plots/{a}-{b}-plot.html"
        urls.append(file_path)
        fig.write_html(file=file_path, include_plotlyjs="cdn")
        # Table for Single Predictor
        # print(MOR_Table)
        # print(urls[0])
        # print(response[0],predictors[i],mean_squared_diff,mean_squared_diff_weighted,urls)

        final_Brute_Force_table = final_Brute_Force_table.append(
            {
                "Predictor_1": a,
                "Predictor_2": b,
                "Mean_Squared_Error": mean_squared_diff,
                "Mean_Squared_Error_Weighted": mean_squared_diff_weighted,
                "URL": urls[0],
            },
            ignore_index=True,
        )
    final_Brute_Force_table = final_Brute_Force_table.sort_values(
        "Mean_Squared_Error_Weighted", ascending=False
    )
    # final_MOR_1D_table = final_MOR_1D_table.reset_index(drop=True)
    final_Brute_Force_table = final_Brute_Force_table.style.set_properties(
        **{"border": "1.3px solid black"}
    ).format({"URL": make_clickable}, escape="html")
    final_Brute_Force_table.to_html("Brute_Force.html")
    # print(Brute_Force_Table.head())
    # print(Brute_Force_Table.shape)


def make_clickable(val):
    return '<a href="{}">{}</a>'.format(val, val)


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


def cat_response_cont_predictor(df, i, response, predictors, table):
    # print(df.columns[i])
    print("Called: cat_response_cont_predictor")
    # print(df[response])
    group_labels = df.iloc[:, -1].unique()
    # print(group_labels)
    # print(df[df[response[0]] == group_labels[0]].iloc[:, 0].values)
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
    # fig_1.show()
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
    # fig_2.show()

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

    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title=f"Variable: {df.columns[i]}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {df.columns[i]}",
        yaxis_title="y",
    )
    # fig.show()
    urls_dist = []
    if not os.path.isdir("Distribution-plots"):
        os.mkdir("Distribution-plots")
    file_path_dist = f"Distribution-plots/{df.columns[i]}-plot.html"
    urls_dist.append(file_path_dist)
    fig_1.write_html(file=file_path_dist, include_plotlyjs="cdn")

    urls_violin = []
    if not os.path.isdir("Violin-plots"):
        os.mkdir("Violin-plots")
    file_path_vio = f"Violin-plots/{df.columns[i]}-plot.html"
    urls_violin.append(file_path_vio)
    fig_2.write_html(file=file_path_vio, include_plotlyjs="cdn")

    urls_scatter = []
    if not os.path.isdir("Scatter-plots"):
        os.mkdir("Scatter-plots")
    file_path_sc = f"Scatter-plots/{df.columns[i]}-plot.html"
    urls_scatter.append(file_path_sc)
    fig.write_html(file=file_path_sc, include_plotlyjs="cdn")

    table.append(
        [
            df.columns[i],
            "Continuous Predictor",
            "Categorical Response",
            t_value,
            p_value,
            file_path_dist,
            file_path_vio,
            file_path_sc,
        ]
    )
    return


def correlation_metrics_cont_cont(X, predictors, continuous):
    corr = X[continuous].corr(method="pearson")

    # plotting the heatmap for correlation
    fig = sns.heatmap(X.corr(), annot=False)
    plt.savefig("Correlation_Matrix.png")
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
    # fig.show()
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


def get_data():

    mydb = mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="believe",
        database="baseball",
    )
    df = pd.read_sql("SELECT * FROM FEATURES", con=mydb)

    # df=pd.read_table('./results/Features.csv')
    '''

  db_user = "root"
  db_pass = "believe"  # pragma: allowlist secret
  db_host = "localhost"
  db_database = "baseball"

  connect_string = (
    f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
  )

  sql_engine = sqlalchemy.create_engine(connect_string)

  query = """SELECT * FROM FEATURES"""

  with sql_engine.begin() as connection:
    df = pd.read_sql_query(text(query), connection)
  '''
    return df


def inspect_data(df):
    # Check data shape
    # print(df['HomeTeamWins'].unique())
    # print(df['HomeTeamWins'].value_counts())
    # print(df.shape)
    # Check Top 5 rows
    # print(df.head())
    # Check datatype
    print(df.dtypes)
    # Check data info
    # print(df.info())
    # Get data summary
    # print(df.describe())
    # Check for NULL values
    print(df.isnull().sum())
    # Check Correlation
    # print(df.corr())


def clean_data(df):
    df["local_date"] = pd.to_datetime(df["local_date"])
    df = df.dropna()
    # print(df.isnull().values.sum())
    # print(df.shape)
    return df


def build_models(df, pr=["local_date"]):
    # Removing predictors with high p values

    new_data = df.loc[:, ~df.columns.isin(pr)]
    print(new_data.columns)
    train = new_data.loc[df["local_date"] < "2010-12-12"]
    test = new_data.loc[df["local_date"] >= "2010-12-12"]
    #
    # print('Ends here')
    X_train = train.loc[:, train.columns != "HomeTeamWins"]
    # print(X_train.dtypes)
    X_test = test.loc[:, test.columns != "HomeTeamWins"]
    y_train = train.loc[:, train.columns == "HomeTeamWins"]
    y_test = test.loc[:, test.columns == "HomeTeamWins"]
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    # print(X_train.dtypes)
    # print(y_train.dtypes)

    # LOGISTIC REGRESSION

    log_reg = LogisticRegression().fit(X_train, y_train)
    # print(log_reg.summary())
    y_pred_LOGIT = log_reg.predict(X_test)

    accuracy_LOGIT = metrics.accuracy_score(y_test, (y_pred_LOGIT))
    F1_Macro_LOGIT = metrics.f1_score(y_test, (y_pred_LOGIT), average="macro")
    F1_Micro_LOGIT = metrics.f1_score(y_test, (y_pred_LOGIT), average="micro")
    F1_Weighted_LOGIT = metrics.f1_score(y_test, (y_pred_LOGIT), average="weighted")
    Precision_Macro_LOGIT = metrics.precision_score(
        y_test, (y_pred_LOGIT), average="macro"
    )
    Precision_Micro_LOGIT = metrics.precision_score(
        y_test, (y_pred_LOGIT), average="micro"
    )
    Precision_Weighted_LOGIT = metrics.precision_score(
        y_test, (y_pred_LOGIT), average="weighted"
    )

    print("Logistic Regression Accuracy = ", accuracy_LOGIT)
    print("F1 Score - Macro", F1_Macro_LOGIT)
    print("F1 Score - Micro", F1_Micro_LOGIT)
    print("F1 Score - Weighted", F1_Weighted_LOGIT)
    print("Precision Score - Macro", Precision_Macro_LOGIT)
    print("Precision Score - Micro", Precision_Micro_LOGIT)
    print("Precision Score - Weighted", Precision_Weighted_LOGIT)
    # print((y_pred_LOGIT))
    print(confusion_matrix(y_test, y_pred_LOGIT))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_LOGIT)

    # plt.show()

    plt.savefig("LR_Confusion_Matrix.png")

    # Decision Tree

    DT_clf = tree.DecisionTreeClassifier(
        random_state=0, max_depth=2, min_samples_split=2
    )
    DT_Fitted = DT_clf.fit(X_train, y_train)
    # print(DT_clf.feature_importances_)
    y_pred_DT = DT_Fitted.predict(X_test)
    accuracy_DT = metrics.accuracy_score(y_test, y_pred_DT)
    F1_Macro_DT = metrics.f1_score(y_test, y_pred_DT, average="macro")
    F1_Micro_DT = metrics.f1_score(y_test, y_pred_DT, average="micro")
    F1_Weighted_DT = metrics.f1_score(y_test, y_pred_DT, average="weighted")
    Precision_Macro_DT = metrics.precision_score(y_test, y_pred_DT, average="macro")
    Precision_Micro_DT = metrics.precision_score(y_test, y_pred_DT, average="micro")
    Precision_Weighted_DT = metrics.precision_score(
        y_test, y_pred_DT, average="weighted"
    )

    print("Decision Tree Accuracy:", accuracy_DT)
    print("F1 Score - Macro", F1_Macro_DT)
    print("F1 Score - Micro", F1_Micro_DT)
    print("F1 Score - Weighted", F1_Weighted_DT)
    print("Precision Score - Macro", Precision_Macro_DT)
    print("Precision Score - Micro", Precision_Micro_DT)
    print("Precision Score - Weighted", Precision_Weighted_DT)
    print(confusion_matrix(y_test, y_pred_DT))
    # plot_confusion_matrix(DT_clf, X_test, y_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_DT)

    plt.savefig("DT_Confusion_Matrix.png")

    # Random Forest

    RF_clf = RandomForestClassifier(max_depth=19, random_state=0)
    RF_Fitted = RF_clf.fit(X_train, y_train)
    y_pred_RF = RF_Fitted.predict(X_test)

    accuracy_RF = metrics.accuracy_score(y_test, y_pred_RF)
    F1_Macro_RF = metrics.f1_score(y_test, y_pred_RF, average="macro")
    F1_Micro_RF = metrics.f1_score(y_test, y_pred_RF, average="micro")
    F1_Weighted_RF = metrics.f1_score(y_test, y_pred_RF, average="weighted")
    Precision_Macro_RF = metrics.precision_score(y_test, y_pred_RF, average="macro")
    Precision_Micro_RF = metrics.precision_score(y_test, y_pred_RF, average="micro")
    Precision_Weighted_RF = metrics.precision_score(
        y_test, y_pred_RF, average="weighted"
    )

    print("Random Forest Accuracy:", accuracy_RF)
    print("Precision Score - Macro", Precision_Macro_RF)
    print("Precision Score - Micro", Precision_Micro_RF)
    print("Precision Score - Weighted", Precision_Weighted_RF)
    print("F1 Score - Macro", F1_Macro_RF)
    print("F1 Score - Micro", F1_Micro_RF)
    print("F1 Score - Weighted", F1_Weighted_RF)
    print(confusion_matrix(y_test, y_pred_RF))
    # plot_confusion_matrix(RF_clf, X_test, y_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_RF)

    plt.savefig("RF_Confusion_Matrix.png")
    importances = RF_clf.feature_importances_
    # print(len(X_train.columns))
    # print(len(importances))

    std = np.std([tree.feature_importances_ for tree in RF_clf.estimators_], axis=0)
    forest_importances_impurity = pd.Series(importances, index=X_train.columns)
    fig, ax = plt.subplots()
    forest_importances_impurity.sort_values(ascending=False).plot.bar()
    ax.set_title("Feature Importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    print(forest_importances_impurity.sort_values(ascending=False))
    plt.savefig("RF_Variable_Importance.png")

    # Gradient Boosting Classifier

    GB_clf = GradientBoostingClassifier(
        learning_rate=0.15, max_features=18, max_depth=6, random_state=0
    )
    GB_Fitted = GB_clf.fit(X_train, y_train)
    y_pred_GB = GB_Fitted.predict(X_test)
    accuracy_GB = metrics.accuracy_score(y_test, y_pred_GB)
    F1_Macro_GB = metrics.f1_score(y_test, y_pred_GB, average="macro")
    F1_Micro_GB = metrics.f1_score(y_test, y_pred_GB, average="micro")
    F1_Weighted_GB = metrics.f1_score(y_test, y_pred_GB, average="weighted")
    Precision_Macro_GB = metrics.precision_score(y_test, y_pred_GB, average="macro")
    Precision_Micro_GB = metrics.precision_score(y_test, y_pred_GB, average="micro")
    Precision_Weighted_GB = metrics.precision_score(
        y_test, y_pred_GB, average="weighted"
    )

    print("Gradient Boosting Accuracy:", accuracy_GB)
    print("F1 Score - Macro", F1_Macro_GB)
    print("F1 Score - Micro", F1_Micro_GB)
    print("F1 Score - Weighted", F1_Weighted_GB)
    print("Precision Score - Macro", Precision_Macro_GB)
    print("Precision Score - Micro", Precision_Micro_GB)
    print("Precision Score - Weighted", Precision_Weighted_GB)
    print(confusion_matrix(y_test, y_pred_GB))
    # plot_confusion_matrix(GB_clf, X_test, y_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_GB)
    plt.savefig("GB_Confusion_Matrix.png")
    # plt.show()

    # Support Vector Classifier

    svc_clf = SVC()
    svc_fitted = svc_clf.fit(X_train, y_train)
    y_pred_svc = svc_fitted.predict(X_test)
    accuracy_SVC = metrics.accuracy_score(y_test, y_pred_svc)
    F1_Macro_SVC = metrics.f1_score(y_test, y_pred_svc, average="macro")
    F1_Micro_SVC = metrics.f1_score(y_test, y_pred_svc, average="micro")
    F1_Weighted_SVC = metrics.f1_score(y_test, y_pred_svc, average="weighted")
    Precision_Macro_SVC = metrics.precision_score(y_test, y_pred_svc, average="macro")
    Precision_Micro_SVC = metrics.precision_score(y_test, y_pred_svc, average="micro")
    Precision_Weighted_SVC = metrics.precision_score(
        y_test, y_pred_svc, average="weighted"
    )

    print("Support Vector Classifier Accuracy:", accuracy_SVC)
    print("F1 Score - Macro", F1_Macro_SVC)
    print("F1 Score - Micro", F1_Micro_SVC)
    print("F1 Score - Weighted", F1_Weighted_SVC)
    print("Precision Score - Macro", Precision_Macro_SVC)
    print("Precision Score - Micro", Precision_Micro_SVC)
    print("Precision Score - Weighted", Precision_Weighted_SVC)
    print(confusion_matrix(y_test, y_pred_svc))
    # plot_confusion_matrix(svc_clf, X_test, y_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svc)
    plt.savefig("SVC_Confusion_Matrix.png")
    # plt.show()

    """
  #Grid Search CV
  param_grid = {'C': [0.1],
                'gamma': [1],
                'kernel': ['rbf']}

  grid_clf = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
  grid_fitted =grid_clf.fit(X_train, y_train)
  y_pred_grid = grid_fitted.predict(X_test)

  accuracy_GRID = metrics.accuracy_score(y_test, y_pred_grid)
  F1_Macro_GRID = metrics.f1_score(y_test, y_pred_grid, average="macro")
  F1_Micro_GRID = metrics.f1_score(y_test, y_pred_grid, average="micro")
  F1_Weighted_GRID = metrics.f1_score(y_test, y_pred_grid, average="weighted")
  Precision_Macro_GRID = metrics.precision_score(y_test, y_pred_grid, average="macro")
  Precision_Micro_GRID = metrics.precision_score(y_test, y_pred_grid, average="micro")
  Precision_Weighted_GRID = metrics.precision_score(y_test, y_pred_grid, average="weighted")

  print("Grid Search Classifier Accuracy:", accuracy_GRID)
  print("F1 Score - Macro", F1_Macro_GRID)
  print("F1 Score - Micro", F1_Micro_GRID)
  print("F1 Score - Weighted", F1_Weighted_GRID)
  print("Precision Score - Macro", Precision_Macro_GRID)
  print("Precision Score - Micro", Precision_Micro_GRID)
  print("Precision Score - Weighted", Precision_Weighted_GRID)
  print(confusion_matrix(y_test, y_pred_grid))
  plot_confusion_matrix(grid_clf, X_test, y_test)
  plt.show()
  """


def find_p_value(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    logistic_model = sm.Logit(y, X)
    results = logistic_model.fit()
    p_values = results.pvalues
    columns = ["Predictors", "P-Values"]
    table = pd.DataFrame([], columns=columns)

    # print(p_values.values)
    table["Predictors"] = X.columns
    table["P-Values"] = p_values.values
    # print(table.head())
    table = table.sort_values("P-Values", ascending=True)
    # final_MOR_1D_table = final_MOR_1D_table.reset_index(drop=True)
    table = table.style.set_properties(**{"border": "1.3px solid black"}).format(
        {"URL": make_clickable}, escape="html"
    )
    table.to_html("P-Values.html")

    return


def main():
    df = get_data()

    # print(df.dtypes)
    inspect_data(df)

    df = clean_data(df)
    new_data = df.copy()
    predictors = [
        "game_id",
        "home_team_id",
        "away_team_id",
        "HT_Batter_Average",
        "AT_Batter_Average",
        "Diff_Batter_Average",
        "HT_Batter_Ground_fly_ball_ratio",
        "AT_Batter_Ground_fly_ball_ratio",
        "Diff_Batter_Ground_fly_ball_ratio",
        "HT_Batter_Plate_App_Strikeout",
        "AT_Batter_Plate_App_Strikeout",
        "Diff_Batter_Plate_App_Strikeout",
        "HT_Batter_strikeout_Walk_ratio",
        "AT_Batter_strikeout_Walk_ratio",
        "Diff_Batter_strikeout_Walk_ratio",
        "HT_Batter_Walk_strikeout_ratio",
        "AT_Batter_Walk_strikeout_ratio",
        "Diff_Batter_Walk_strikeout_ratio",
        "HT_Batting_BABIP",
        "AT_Batting_BABIP",
        "Diff_Batting_BABIP",
        "HT_Home_runs_per_hit",
        "AT_Home_runs_per_hit",
        "Diff_Home_runs_per_hit",
        "HT_BB9",
        "AT_BB9",
        "Diff_BB9",
        "HT_H9",
        "AT_H9",
        "Diff_H9",
        "HT_HR9",
        "AT_HR9",
        "DIff_HR9",
        "HT_S9",
        "AT_S9",
        "Diff_S9",
        "HT_WHIP",
        "AT_WHIP",
        "Diff_WHIP",
        "Home_Team_Strikouts_Normal",
        "Away_Team_Strikouts_Normal",
        "Diff_Strikouts_Normal",
        "Home_Team_PlateApperance_Normal",
        "Away_Team_PlateApperance_Normal",
        "Diff_Team_PlateApperance_Normal",
        "Home_Team_Single_Normal",
        "Away_Team_Single_Normal",
        "Diff_Single_Normal",
        "Home_Team_Double_Normal",
        "Away_Team_Double_Normal",
        "Diff_Team_Double_Normal",
        "Home_Team_Triple_Normal",
        "Away_Team_Triple_Normal",
        "Diff_Triple_Normal",
        "Home_Team_Slugging_Percentage_Normal",
        "Away_Team_Slugging_Percentage_Normal",
        "Diff_Slugging_Percentage_Normal",
        "Home_Team_Batting_Average_Normal",
        "Away_Team_Batting_Average_Normal",
        "Diff_Batting_Average_Normal",
        "Home_Team_Walk_strikeout_ratio_Normal",
        "Away_Team_Walk_strikeout_ratio_Normal",
        "Diff_Walk_strikeout_ratio_Normal",
        "Home_Team_Ground_fly_ball_ratio_Normal",
        "Away_Team_Ground_fly_ball_ratio_Normal",
        "Diff_Ground_fly_ball_ratio_Normal",
        "Home_Team_Intentional_Walk_Normal",
        "Away_Team_Intentional_Walk_Normal",
        "Diff_Intentional_Walk_Normal",
        "Home_Team_At_bats_per_home_run_Normal",
        "Away_Team_At_bats_per_home_run_Normal",
        "Diff_At_bats_per_home_run_Normal",
        "Home_Team_Home_runs_per_hit_Normal",
        "Away_Team_Home_runs_per_hit_Normal",
        "Diff_Home_runs_per_hit_Normal",
        #'Home_Team_Pythagorean_Win_Ratio', 'Home_Team_Pythagorean_Win_Ratio'
    ]
    # print(len(predictors))
    response = ["HomeTeamWins"]
    column_name, column_type, categorical, continuous = split_predictors(df, predictors)
    result_response = "CAT_RES"
    table = []
    df = df.loc[:, ~df.columns.isin(["local_date"])]
    df1 = df.copy()
    # print('First Df1')
    # print(df1.dtypes)
    # Variable Evaluation

    for i in range(0, len(df.columns) - 1):
        # result_response = check_response(df,i,response)
        result_predictor = check_predictor(df, i, predictors)
        if result_response == "CAT_RES":
            df[response] = df[response].astype(str)
            if result_predictor == "CAT_PRED":
                df.iloc[:, i] = df.iloc[:, i].astype("category").cat.codes
                # cat_response_cat_predictor(df, i, response, predictors, table)
                print("Entered cat_response_cat_predictor")
            else:
                cat_response_cont_predictor(df, i, response, predictors, table)
        else:
            if result_predictor == "CAT_PRED":
                df.iloc[:, i] = df.iloc[:, i].astype("category").cat.codes
                # cont_response_cat_pred(df, i, response, predictors, table)
                print("Entered cont_response_cat_pred")
            else:
                # cont_response_cont_predictor(df, i, response, predictors, table)
                print("Entered cont_response_cont_predictor")

    # print(df.dtypes)
    # Correlation
    X = df[predictors]
    # y = df[response[0]]
    print("Starting correlation_metrics_cont_cont ")
    correlation_metrics_cont_cont(X, predictors, continuous)

    # print('df1 right before MOR')
    # print(df1.dtypes)
    # Mean of Response 1D
    print("Starting MeanOfResponse ")
    MeanOfResponse(df1, predictors, response)
    print("Starting Brute_Force_cont_cont ")
    Brute_Force_cont_cont(df1, predictors, continuous, response)

    build_models(new_data)

    # removing a few insignificant features i.e, p<70%

    build_models(
        new_data,
        [
            "local_date",
            "AT_WHIP",
            "HT_WHIP",
            "HT_BB9",
            "AT_BB9",
            "HT_H9",
            "AT_H9",
            "HT_Batter_strikeout_Walk_ratio",
            "HT_Batter_Ground_fly_ball_ratio",
            "AT_Batter_Average",
            "Home_Team_Strikouts_Normal",
            "Away_Team_Intentional_Walk_Normal",
            "HT_Batter_Walk_strikeout_ratio",
            "Diff_Batter_Ground_fly_ball_ratio",
            "Diff_S9",
            "Diff_H9",
            "Diff_Batter_Plate_App_Strikeout",
            "AT_Batting_BABIP",
        ],
    )

    # removing a few more insignificant variables p<=50%
    build_models(
        new_data,
        [
            "local_date",
            "AT_WHIP",
            "HT_WHIP",
            "HT_BB9",
            "AT_BB9",
            "HT_H9",
            "AT_H9",
            "HT_Batter_strikeout_Walk_ratio",
            "HT_Batter_Ground_fly_ball_ratio",
            "AT_Batter_Average",
            "Home_Team_Strikouts_Normal",
            "Away_Team_Intentional_Walk_Normal",
            "HT_Batter_Walk_strikeout_ratio",
            "Diff_Batter_Ground_fly_ball_ratio",
            "Diff_S9",
            "Diff_H9",
            "Diff_Batter_Plate_App_Strikeout",
            "AT_Batting_BABIP",
            "Diff_Triple_Normal",
            "Home_Team_At_bats_per_home_run_Normal",
            "HT_Home_runs_per_hit",
            "HT_S9",
            "Diff_Single_Normal",
            "Diff_BB9",
            "AT_Batter_strikeout_Walk_ratio",
            "Diff_Slugging_Percentage_Normal",
            "Away_Team_Walk_strikeout_ratio_Normal",
            "Diff_Batter_Walk_strikeout_ratio",
            "Diff_Strikouts_Normal",
        ],
    )

    # removing a few more predictors based on RF importance
    build_models(
        new_data,
        [
            "local_date",
            "AT_WHIP",
            "HT_WHIP",
            "HT_BB9",
            "AT_BB9",
            "HT_H9",
            "AT_H9",
            "HT_Batter_strikeout_Walk_ratio",
            "HT_Batter_Ground_fly_ball_ratio",
            "AT_Batter_Average",
            "Home_Team_Strikouts_Normal",
            "Away_Team_Intentional_Walk_Normal",
            "HT_Batter_Walk_strikeout_ratio",
            "Diff_Batter_Ground_fly_ball_ratio",
            "Diff_S9",
            "Diff_H9",
            "Diff_Batter_Plate_App_Strikeout",
            "AT_Batting_BABIP",
            "Diff_Triple_Normal",
            "Home_Team_At_bats_per_home_run_Normal",
            "HT_Home_runs_per_hit",
            "HT_S9",
            "Diff_Single_Normal",
            "Diff_BB9",
            "AT_Batter_strikeout_Walk_ratio",
            "Diff_Slugging_Percentage_Normal",
            "Away_Team_Walk_strikeout_ratio_Normal",
            "Diff_Batter_Walk_strikeout_ratio",
            "Diff_Strikouts_Normal",
            "Home_Team_Triple_Normal",
            "Diff_Intentional_Walk_Normal",
            "Away_Team_Triple_Normal",
            "Diff_Batter_Average",
            "Diff_Team_Double_Normal",
            "HT_Batter_Average",
            "Diff_Home_runs_per_hit",
            "Diff_Ground_fly_ball_ratio_Normal",
            "Away_Team_Ground_fly_ball_ratio_Normal",
            "AT_Home_runs_per_hit",
            "Home_Team_Ground_fly_ball_ratio_Normal",
            "AT_Batter_Ground_fly_ball_ratio",
            "AT_Batter_Walk_strikeout_ratio",
        ],
    )
    find_p_value(df1)


if __name__ == "__main__":
    sys.exit(main())
