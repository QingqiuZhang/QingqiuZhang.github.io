# -*- coding         : utf-8 -*-
# @Author            : Qingqiu Zhang
# @Date              : 2022-03-09 20: 43: 51
# @Last Modified by  : Qingqiu Zhang
# @Last Modified time: 2022-04-09 16: 38: 34
"""
This is a demonstration for a simple data analysis task!
It went through the eduactional info in USA.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("nces-ed-attainment.csv")
df.head()

df = df.replace("---", np.nan)
df.info()

race = [
    "Total",
    "Hispanic",
    "Asian",
    "Pacific Islander",
    "American Indian/Alaska Native",
    "Two or more races",
]
df[race] = df[race].apply(pd.to_numeric)
df.info()

describe = df.describe(include="all").T
print(describe)


cormat = df[race].corr().round(2)
sns.heatmap(cormat, annot=True)


f, axes = plt.subplots(4, 2, sharey=False, figsize=(15, 12))
for ind, val in enumerate(race):
    sns.histplot(
        data      = df,
        x         = val,
        hue       = "Min degree",
        label     = "100% Equities",
        kde       = True,
        stat      = "density",
        linewidth = 0,
        ax        = axes[ind // 2, ind % 2],
        bins      = 200,
    ).set(title=val)
f.tight_layout(pad=3.0)
plt.show()


f, axes = plt.subplots(4, 2, sharey=False, figsize=(14, 14))
for ind, val in enumerate(race):
    sns.lineplot(
        data = df, x = "Year", y = val, hue = "Min degree", ci = None, ax = axes[ind // 2, ind % 2]
    )


def completion_bet_years(dataframe, year1, year2, sex):
    """ Return percent of different degrees completed between year1 and year2 for Sex==sex.

    Args                 :
    dataframe (DataFrame): A dataframe containing the needed data
    year1     (int)      : Year number: the  earlier one
    year2     (int)      : Year number: the  later one
    sex       (str)      : Gender

    Returns  :
    DataFrame: The percent of degrees completed between year1 and year2 for Sex==sex.
    """
    result = dataframe.loc[
        (dataframe["Sex"] == sex)
        & (dataframe["Year"] >= year1)
        & (dataframe["Year"] < year2)
    ]
    if result.shape[0] == 0:
        return None
    else:
        return result


print(completion_bet_years(df, 1920, 1941, "A"))


def compare_bachelors_in_year(dataframe, year):
    """ Return the percentages for women vs men having earned a Bachelor’s Degree in Year==year

    Args            :
    df   (Dataframe): A dataframe containing the needed data
    year (int)      : The year in which you want to know the info

    Returns:
    tuple  : A tuple returns the percentage of Bachelor’s Degree male and femal in Year==year.
    """
    women = dataframe[
        (dataframe["Sex"] == "F")
        & (dataframe["Min degree"] == "bachelor's")
        & (dataframe["Year"] == year)
    ].iloc[0]["Total"]
    men = dataframe[
        (dataframe["Sex"] == "M")
        & (dataframe["Min degree"] == "bachelor's")
        & (dataframe["Year"] == year)
    ].iloc[0]["Total"]
    return (f"{men} % for men", f"{women} % for women")


compare_bachelors_in_year(df, 2010)


def top_2_2000s(dataframe):
    """Return the two most common educational attainment between 2000-2010.

    Args                 :
    DataFrame (DataFrame): A dataframe containing the needed data

    Returns:
    list   : A list returns the two most common educational attainment between 2000-2010.
    """
    df_2000s = dataframe.loc[
        (dataframe["Year"] >= 2000)
        & (dataframe["Year"] <= 2010)
        & (dataframe["Sex"] == "A")
    ][["Total", "Min degree"]]
    mean_percent_edu = (
        df_2000s.groupby("Min degree").mean().sort_values(by="Total", ascending=False)
    )
    level1 = mean_percent_edu.iloc[0]["Total"].round(2)
    level2 = mean_percent_edu.iloc[1]["Total"].round(2)
    edu1   = mean_percent_edu.index[0]
    edu2   = mean_percent_edu.index[1]
    return [f"#1 level, {level1} % of {edu1}", f"#2 level, {level2} % of {edu2}"]


top_2_2000s(df)
