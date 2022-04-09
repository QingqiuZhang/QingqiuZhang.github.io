# -*- coding: utf-8 -*-
# @Author: Qingqiu Zhang
# @Date:   2022-04-09 16:55:56
# @Last Modified by:   Qingqiu Zhang
# @Last Modified time: 2022-04-09 17:22:57
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from dmba import adjusted_r2_score

day = pd.read_csv("day.csv")
day.head()

result = day[["mnth", "casual", "registered", "cnt"]].groupby(["mnth"]).mean()
result = (
    result.stack()
    .reset_index()
    .set_index("mnth")
    .rename(columns={"level_1": "cat", 0: "people per day"})
)
cat    = ["season", "yr", "holiday", "weekday", "workingday", "weathersit"]

sns.set(style ="ticks", palette="pastel")
f, axes = plt.subplots(3, 3, sharey=False, figsize=(15, 12))
ax      = plt.subplot2grid((3, 3), (0, 0), colspan=3)
sns.barplot(x=result.index, y="people per day", data=result, hue="cat", ax=ax)
for ind, val in enumerate(cat):
    result = round(day[[val, "casual", "registered", "cnt"]].groupby([val]).mean())
    result = (
        result.stack()
        .reset_index()
        .set_index(val)
        .rename(columns={"level_1": "cat", 0: "people per day"})
    )
    sns.barplot(
        x    =result.index,
        y    ="people per day",
        data = result,
        hue  = "cat",
        ax   =axes[ind // 3 + 1, ind % 3],
    )
f.tight_layout(pad=3.0)
plt.show()


result = day[["casual", "registered", "cnt"]].set_index(day["mnth"])
result = (
    result.stack()
    .reset_index()
    .set_index("mnth")
    .rename(columns={"level_1": "cat", 0: "people"})
)

f, axes = plt.subplots(3, 3, sharey=False, figsize=(20, 12))
ax      = plt.subplot2grid((3, 3), (0, 0), colspan=3)
sns.violinplot(x=result.index, y="people", hue="cat", data=result, cut=0, ax=ax)
for ind, val in enumerate(cat):
    result = day[["casual", "registered", "cnt"]].set_index(day[val])
    result = (
        result.stack()
        .reset_index()
        .set_index(val)
        .rename(columns={"level_1": "cat", 0: "people"})
    )
    sns.violinplot(
        x    = result.index,
        y    = "people",
        hue  = "cat",
        data = result,
        cut  = 0,
        ax   = axes[ind // 3 + 1, ind % 3],
    )
f.tight_layout(pad=3.0)
plt.show()

day[cat] = day[cat].apply(lambda x: x.astype("category"))
dummies  = pd.get_dummies(day[cat], drop_first=True)
print(dummies.shape)

conti_predictors = ["temp", "atemp", "hum", "windspeed"]
print(day[conti_predictors].shape)

X = pd.concat([day[conti_predictors], dummies], axis=1)
y = day.iloc[:, 13:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
sc = StandardScaler()
X_train[conti_predictors] = sc.fit_transform(X_train[conti_predictors])
X_test[conti_predictors]  = sc.transform(X_test[conti_predictors])

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_pred = pd.DataFrame(y_pred, columns=["casual", "registered", "cnt"]).astype(int)
print(y_pred)
print(adjusted_r2_score(y_test, y_pred, lr))

for i in range(y_pred.shape[0]):
    if y_pred["casual"].iloc[i] < 0:
        y_pred["casual"].iloc[i] = 0
    if y_pred["registered"].iloc[i] < 0:
        y_pred["registered"].iloc[i] = 0
        y_pred["cnt"].iloc[i] = y_pred["casual"].iloc[i] + y_pred["registered"].iloc[i]
print(y_pred)
print(adjusted_r2_score(y_test, y_pred, lr))
