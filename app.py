from flask import Flask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import os
from pandas_flavor import register_dataframe_method,register_series_method
from IPython.core.display import display, HTML
from matplotlib.ticker import MaxNLocator
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask, redirect, url_for, render_template, request, flash
app = Flask(__name__)
from io import BytesIO
import base64

@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')

@app.route('/result')
def run():
    def plot_histogram():
        img = BytesIO()
        fig =  histplt(df.continuous_features_df(thresold=50), bins= 20, x_size=40, y_size=20)
        fig.savefig(img, format='png')
        fig.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        return plot_url

    def plot_bar():
        img = BytesIO()
        feat = df.discrete_features_cols(thresold=50)
        feat.append("Income")

        fig = barplt(df[feat], y="Income", x_size=40)
        fig.savefig(img, format='png')
        fig.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        return plot_url

    def plot_count():
        img = BytesIO()
        fig = countplt(df.discrete_features_df(thresold=50), y_size=30)
        fig.savefig(img, format='png')
        fig.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        return plot_url

    def plot_violin():
        img = BytesIO()
        feat = df.discrete_features_cols(thresold=50)
        feat.append("Income")
        fig = violinplt(df[feat], y = "Income", ncols=2, x_size=30, y_size=50,)
        fig.savefig(img, format='png')
        fig.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        return plot_url


    return render_template('result.html', plot_url=plot_histogram(),plot_bar = plot_bar(),plot_count = plot_count(), plot_violin = plot_violin())

@app.route('/resul')
def result():
    img = BytesIO()
    feat = df.discrete_features_cols(thresold=50)
    feat.append("Income")

    fig = barplt(df[feat], y="Income", x_size=40)
    fig.savefig(img, format='png')
    fig.close()
    img.seek(0)
    plot_brr = base64.b64encode(img.getvalue()).decode('utf8')
    return render_template('plot.html', plot_brr = plot_brr)

@app.route('/interactive')
def interactive():
    return render_template('interactive.html')


@app.route('/histogram.png')
def plot_png():
    fig = create_histogram()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


if __name__ == '__main__':
    app.run()




for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


@register_dataframe_method
def missing(df):
    tmp = sorted(
        [(col, str(df[col].dtypes), df[col].isna().sum(), np.round(df[col].isna().sum() / len(df) * 100, 2)) for col in
         df.columns if df[col].isna().sum() != 0],
        key=lambda x: x[2], reverse=True)

    return pd.DataFrame(tmp).rename({0: "Feature", 1: "dtype", 2: "count", 3: "percent"}, axis=1)


@register_dataframe_method
def get_numeric_df(df):
    return df.select_dtypes(np.number)


@register_dataframe_method
def get_numeric_cols(df):
    return list(df.select_dtypes(np.number).columns)


@register_dataframe_method
def discrete_features_cols(df, thresold):
    #     thresold in number of unique values
    return [feature for feature in df.columns if len(df[feature].unique()) < thresold]


@register_dataframe_method
def discrete_features_df(df, thresold):
    #     thresold in number of unique values
    return df[discrete_features_cols(df=df, thresold=thresold)]


@register_dataframe_method
def continuous_features_cols(df, thresold):
    #     thresold in number of unique values
    return [feature for feature in df.columns if len(df[feature].unique()) >= thresold]


@register_dataframe_method
def continuous_features_df(df, thresold):
    #     thresold in number of unique values
    return df[continuous_features_cols(df=df, thresold=thresold)]


@register_dataframe_method
def dtypes_of_cols(df):
    return pd.DataFrame(df.dtypes).reset_index().rename(columns={'index': "Columns", 0: "dtype"})


@register_dataframe_method
def describe_discrete_cols(df, thresold, ascending=True):
    values = pd.DataFrame()

    for col in df.discrete_features_cols(thresold=thresold):
        values[col] = [df[col].unique(), df[col].nunique()]

    return values.transpose().sort_values(by=1, ascending=ascending).rename({0: "Values", 1: "cardinality"}, axis=1)


@register_series_method
def IQR_range(df):
    if isinstance(df, pd.Series):
        Q3 = np.quantile(df, 0.75)
        Q1 = np.quantile(df, 0.25)
        IQR = Q3 - Q1

        lower_range = Q1 - 1.5 * IQR
        upper_range = Q3 + 1.5 * IQR

        return (lower_range, upper_range)
    else:
        assert False, "df must be of type pandas.Series"


@register_dataframe_method
def IQR_range(df):
    if isinstance(df, pd.DataFrame):
        cols = df.get_numeric_cols()
        features = {}
        for i in cols:
            Q3 = np.quantile(df[i], 0.75)
            Q1 = np.quantile(df[i], 0.25)
            IQR = Q3 - Q1

            lower_range = Q1 - 1.5 * IQR
            upper_range = Q3 + 1.5 * IQR

            features[i] = (lower_range, upper_range)

        return pd.DataFrame.from_dict(features, orient='index').rename({0: 'IQR_Low', 1: 'IQR_High'}, axis=1)
    else:
        assert False, "df must be of type pandas.DataFrame"


@register_dataframe_method
def compare_cols(df, l_feat, r_feat, percent=False):
    #     [L_feat] {R_feat1: agg1, R_feat2: agg2}

    if percent:

        comp = []
        for key, val in zip(r_feat, r_feat.values()):
            tmp = pd.DataFrame()
            tmp[key + " " + val] = df.groupby(l_feat, sort=True).agg({key: val})
            tmp[key + " %"] = tmp.groupby(level=0).apply(lambda x: np.round(100 * x / float(x.sum()), 2))

            comp.append(tmp)

        return comp

    else:
        comp = []
        for key, val in zip(r_feat, r_feat.values()):
            tmp = pd.DataFrame()
            tmp[key + " " + val] = df.groupby(l_feat, sort=True).agg({key: val})
            comp.append(tmp)

        return comp


@register_series_method
def IQR_percent(df):
    if isinstance(df, pd.Series):

        lower_range, upper_range = df.IQR_range()

        length = len(df)
        return np.round((length - df.between(lower_range, upper_range).sum()) / length * 100, 2)
    else:
        assert False, "df must be of type pandas.Series"


@register_dataframe_method
def IQR_percent(df):
    if isinstance(df, pd.DataFrame):
        cols = df.get_numeric_cols()
        features = {}
        for i in cols:
            lower_range, upper_range = df[i].IQR_range()

            length = len(df[i])
            tmp = np.round((length - df[i].between(lower_range, upper_range).sum()) / length * 100, 2)
            if tmp != 0:
                features[i] = tmp
        #             features[i] = IQR_percent(df[i])

        return pd.DataFrame.from_dict(features, orient='index').rename({0: 'Outlier percent'}, axis=1)
    else:
        assert False, "df must be of type pandas.DataFrame"


@register_dataframe_method
def drop_row_outlier(df, cols, inplace=False):
    #     init empty index
    indices = pd.Series(np.zeros(len(df), dtype=bool), index=df.index)

    for col in cols:
        low, top = df[col].IQR_range()
        indices |= (df[col] > top) | (df[col] < low)

    return df.drop(df[indices].index, inplace=inplace)


@register_series_method
def drop_row_outlier(df, inplace=False):
    #     init empty index

    low, top = df.IQR_range()

    return df.drop(df[(df > top) | (df < low)].index, inplace=inplace)


@register_dataframe_method
def about(df):
    display(HTML('<h1 style="color:green"> <b> Shape of data </b> </h1>'))
    print(df.shape)

    display(HTML('<h1 style="color:green"> <b> Datatypes in data </b> </h1> '))
    print(df.dtypes.value_counts(ascending=False))

    display(HTML('<h1 style="color:green"> <b> dtypes of columns </b> </h1> '))
    display(df.dtypes_of_cols())

    display(HTML('<h1 style="color:green"> <b> Percentage of missing values </b> </h1> '))
    tmp = missing(df)
    display(tmp) if len(tmp) != 0 else display(HTML("<h2> <b> None <b> </h2>"))

    display(HTML('<h1 style="color:green"> <b> Data description </b> </h1> '))
    display(df.describe().T)

    display(HTML('<h1 style="color:green"> <b> Outlier Percentage(IQR) </b> </h1> '))
    tmp = df.IQR_percent()
    display(tmp) if len(tmp) != 0 else display(HTML("<h2> <b> None <b> </h2>"))

    display(HTML('<h1 style="color:green"> <b> Example of data </b> </h1> '))
    display(df.head())


#Plotting Methods


sns.set(style="darkgrid", font_scale=1.3)
plt.rcParams['figure.dpi'] = 100

from matplotlib.ticker import MaxNLocator


def srt_reg(y, df, x_size=20, y_size=20, *args, **kwargs):
    ncols = 3
    nrows = int(np.ceil(df.shape[1] / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(x_size, y_size))
    axes = axes.flatten()

    for i, j in zip(df.columns, axes):
        sns.regplot(x=i,
                    y=y,
                    data=df,
                    ax=j,
                    order=3,
                    ci=None,
                    color='#e74c3c',
                    line_kws={'color': 'black'},
                    scatter_kws={'alpha': 0.4},
                    *args, **kwargs)
        j.tick_params(labelrotation=45)
        j.yaxis.set_major_locator(MaxNLocator(nbins=10))

        plt.tight_layout()


def srt_box(y, df, *args, **kwargs):
    fig, axes = plt.subplots(19, 3, figsize=(30, 30))
    axes = axes.flatten()

    for i, j in zip(df.columns, axes):
        sortd = df.groupby([i])[y].median().sort_values(ascending=False)
        sns.boxplot(x=i,
                    y=y,
                    data=df,
                    palette='plasma',
                    order=sortd.index,
                    ax=j,
                    *args, **kwargs)
        j.tick_params(labelrotation=45)
        j.yaxis.set_major_locator(MaxNLocator(nbins=18))

        plt.tight_layout()


def histplt(df, ncols=3, x_size=30, y_size=30, *args, **kwargs):
    if len(df.shape) == 1:
        fig, ax = plt.subplots(figsize=(x_size, y_size))
        sns.histplot(x=df, ax=ax, *args, **kwargs)
        [ax.bar_label(tmp) for tmp in ax.containers]

        ax.tick_params(labelrotation=45)
    #         plt.tight_layout()

    else:

        #         ncols = 3
        nrows = int(np.ceil(df.shape[1] / ncols))

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(x_size, y_size)
                                 )
        axes = axes.flatten()

        for i, j in zip(df.columns, axes):
            sns.histplot(data=df, x=i, ax=j, *args, **kwargs)
            j.tick_params(labelrotation=45)
            [j.bar_label(tmp) for tmp in j.containers]
            #         j.yaxis.set_major_locator(MaxNLocator(nbins=18))

            plt.tight_layout()
        return plt

def countplt(df, ncols=3, x_size=30, y_size=30, *args, **kwargs):
    if len(df.shape) == 1:
        fig, ax = plt.subplots(figsize=(x_size, y_size))
        sns.countplot(x=df, ax=ax, *args, **kwargs)
        [ax.bar_label(tmp) for tmp in ax.containers]

        ax.tick_params(labelrotation=45)
    #         plt.tight_layout()

    else:

        #         ncols = 3
        nrows = int(np.ceil(df.shape[1] / ncols))

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(x_size, y_size)
                                 )
        axes = axes.flatten()

        for i, j in zip(df.columns, axes):
            sns.countplot(data=df, x=i, ax=j, *args, **kwargs)
            j.tick_params(labelrotation=45)
            [j.bar_label(tmp) for tmp in j.containers]
            #         j.yaxis.set_major_locator(MaxNLocator(nbins=18))

            plt.tight_layout()
        return plt


def barplt(df, y, x_size=30, y_size=30, *args, **kwargs):
    ncols = 3
    nrows = int(np.ceil(df.shape[1] / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(x_size, y_size)
                             )
    axes = axes.flatten()

    for i, j in zip(df.columns, axes):

        if i == y:
            continue

        sns.barplot(data=df,
                    x=i,
                    y=y,
                    ax=j, *args, **kwargs)

        j.tick_params(labelrotation=45)
        [j.bar_label(tmp) for tmp in j.containers]
        #         j.yaxis.set_major_locator(MaxNLocator(nbins=18))

        plt.tight_layout()


    return plt


def violinplt(df, y, ncols=3, x_size=30, y_size=30, x_scale="linear", y_scale="linear", *args, **kwargs):
    nrows = int(np.ceil(df.shape[1] / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(x_size, y_size)
                             )
    axes = axes.flatten()

    if df[y].dtype == 'O':

        for i, j in zip(df.columns, axes):

            if i == y:
                continue

            sns.violinplot(data=df,
                           x=y,
                           y=i,
                           ax=j, *args, **kwargs)

            lower_range, upper_range = df[i].IQR_range()
            outliers = df[(df[i] > upper_range) | (df[i] < lower_range)][i]
            sns.scatterplot(y=outliers, x=0, marker='D', color='crimson', ax=j)
            j.tick_params(labelrotation=45)

            #         j.yaxis.set_major_locator(MaxNLocator(nbins=18))

            plt.tight_layout()


    else:

        for i, j in zip(df.columns, axes):

            if i == y:
                continue

            g = sns.violinplot(data=df,
                               x=i,
                               y=y,
                               ax=j, *args, **kwargs)
            g.set_xscale(x_scale)
            g.set_yscale(y_scale)
            j.tick_params(labelrotation=45)

            #         j.yaxis.set_major_locator(MaxNLocator(nbins=18))

            plt.tight_layout()
    return plt



import scipy.stats as stats





#main method
df = pd.read_csv("static/marketing_campaign.csv", sep="\t")
def code():
  #  df = pd.read_csv("static/marketing_campaign.csv", sep="\t")
    df.drop("ID", axis=1, inplace=True)
    df.about()

    df.dropna(axis=0, inplace=True)
    df.drop("Dt_Customer", axis=1, inplace=True)

    pd.DataFrame(df.nunique()).sort_values(0).rename({0: "Unique Values"}, axis=1)

    df.drop(["Z_CostContact", "Z_Revenue"], axis=1, inplace=True)
    df.continuous_features_cols(thresold=50)
    cols = ['Year_Birth', 'Income']

    df.drop_row_outlier(cols=cols, inplace=True)

code()

def create_histogram():
    return histplt(df.continuous_features_df(thresold=50), bins= 20, x_size=40, y_size=20)