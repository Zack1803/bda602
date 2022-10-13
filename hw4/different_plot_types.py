import sys

import numpy
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix


def cont_resp_cat_predictor():
    n = 200

    # Add histogram data
    x1 = numpy.random.randn(n) - 2
    x2 = numpy.random.randn(n)
    x3 = numpy.random.randn(n) + 2
    x4 = numpy.random.randn(n) + 4

    # Group data together
    hist_data = [x1, x2, x3, x4]

    group_labels = ["Group 1", "Group 2", "Group 3", "Group 4"]

    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
    fig_1.update_layout(
        title="Continuous Response by Categorical Predictor",
        xaxis_title="Response",
        yaxis_title="Distribution",
    )
    fig_1.show()
    fig_1.write_html(
        file="../../../plots/lecture_6_cont_response_cat_predictor_dist_plot.html",
        include_plotlyjs="cdn",
    )

    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig_2.add_trace(
            go.Violin(
                x=numpy.repeat(curr_group, n),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title="Continuous Response by Categorical Predictor",
        xaxis_title="Groupings",
        yaxis_title="Response",
    )
    fig_2.show()
    fig_2.write_html(
        file="../../../plots/lecture_6_cont_response_cat_predictor_violin_plot.html",
        include_plotlyjs="cdn",
    )
    return


def cat_resp_cont_predictor():
    n = 200

    # Add histogram data
    x1 = numpy.random.randn(n) - 2
    x3 = numpy.random.randn(n) + 2

    # Group data together
    hist_data = [x1, x3]

    group_labels = ["Response = 0", "Response = 1"]

    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
    fig_1.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )
    fig_1.show()
    fig_1.write_html(
        file="../../../plots/lecture_6_cat_response_cont_predictor_dist_plot.html",
        include_plotlyjs="cdn",
    )

    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig_2.add_trace(
            go.Violin(
                x=numpy.repeat(curr_group, n),
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
    fig_2.write_html(
        file="../../../plots/lecture_6_cat_response_cont_predictor_violin_plot.html",
        include_plotlyjs="cdn",
    )
    return


def cat_response_cat_predictor():
    n = 200
    x = numpy.random.uniform(0, 1, n)
    y = numpy.random.uniform(0, 1, n)

    x_2 = [1 if abs(x_) > 0.5 else 0 for x_ in x]
    y_2 = [1 if abs(y_) > 0.5 else 0 for y_ in y]

    conf_matrix = confusion_matrix(x_2, y_2)

    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    fig_no_relationship.update_layout(
        title="Categorical Predictor by Categorical Response (without relationship)",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_no_relationship.show()
    fig_no_relationship.write_html(
        file="../../../plots/lecture_6_cat_response_cat_predictor_heat_map_no_relation.html",
        include_plotlyjs="cdn",
    )

    x = numpy.random.randn(n)
    y = x + numpy.random.randn(n)

    x_2 = [1 if abs(x_) > 1.5 else 0 for x_ in x]
    y_2 = [1 if abs(y_) > 1.5 else 0 for y_ in y]

    conf_matrix = confusion_matrix(x_2, y_2)

    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    fig_no_relationship.update_layout(
        title="Categorical Predictor by Categorical Response (with relationship)",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_no_relationship.show()
    fig_no_relationship.write_html(
        file="../../../plots/lecture_6_cat_response_cat_predictor_heat_map_yes_relation.html",
        include_plotlyjs="cdn",
    )
    return


def cont_response_cont_predictor():
    n = 200
    x = numpy.random.randn(n) - 2
    y = x + numpy.random.randn(n) / 5

    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title="Continuous Response by Continuous Predictor",
        xaxis_title="Predictor",
        yaxis_title="Response",
    )
    fig.show()
    fig.write_html(
        file="../../../plots/lecture_6_cont_response_cont_predictor_scatter_plot.html",
        include_plotlyjs="cdn",
    )

    return


def main():
    numpy.random.seed(seed=1234)
    cont_resp_cat_predictor()
    cat_resp_cont_predictor()
    cat_response_cat_predictor()
    cont_response_cont_predictor()
    return


if __name__ == "__main__":
    sys.exit(main())
