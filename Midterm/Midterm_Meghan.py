import os
import sys
import warnings
from itertools import combinations

import pandas as pd
import plotly.express as px
from cat_correlation import cat_cont_correlation_ratio, cat_correlation
from plotly import figure_factory as ff
from plotly import graph_objects as go

# from sklearn import datasets

# , product


warnings.filterwarnings("ignore")


def get_data(dataset_name, predictors, response):
    df = pd.read_csv(dataset_name)
    X = df[predictors]
    y = df[response]
    return df, X, y


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


def encode_categorical_predictors(X, predictors, categorical):
    for cat_col in categorical:
        X[cat_col] = X[cat_col].astype("category").cat.codes.astype("int64")

    return X


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


def make_clickable(val):
    return '<a href="{}">{}</a>'.format(val, val)


def mean_of_response_cont_cont(X, continuous, response):
    Predictor1 = []
    Predictor2 = []
    Unweighted_Mean_of_Response = []
    Weighted_Mean_of_Response = []
    urls = []
    bin_size = 10
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
                print(joined_bin)

                grouped_bin = joined_bin.groupby(bin_columns)
                bin_mean = grouped_bin.mean().unstack()
                print(bin_mean.columns)

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
    brute_cont_cont_table = table.to_html("Brute_Force_Continuous_Continuous.html")
    print(brute_cont_cont_table)


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


def main():

    dataset_name = "Heart.csv"
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
    response = "HeartDisease"
    """
    diabetes = datasets.load_diabetes(as_frame=True)
    data = diabetes['frame']
    predictors = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    response = 'target'
    iris = datasets.load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
    predictos = iris['fest']
    X = data[predictors]
    y = data[response]
    """
    # Get Data Based on dataset name, predictors and response
    data, X, y = get_data(dataset_name, predictors, response)

    # Split the data into categorical and continuous
    column_name, column_type, categorical, continuous = split_predictors(X, predictors)

    # Encode the categorical variables
    # X = encode_categorical_predictors(X,predictors,categorical)
    # print(X.head())

    # Compute the correlation metrics for predictors
    # correlation_metrics_cont_cont(X, predictors, continuous)
    # correlation_metrics_cat_cat(X, predictors, categorical)
    # correlation_metrics_cat_cont(X, predictors, continuous, categorical)
    # print(data.dtypes)
    # Mean of Response
    # evaluate_data(data,continuous,categorical,response)
    print(continuous)
    print(data.columns)
    mean_of_response_cont_cont(data, continuous, response)
    # mean_of_response_cat_cat(data, categorical, response)
    # mean_of_response_cat_cont(data, categorical, continuous,response)
    # print(X.dtypes)


if __name__ == "__main__":
    sys.exit(main())
