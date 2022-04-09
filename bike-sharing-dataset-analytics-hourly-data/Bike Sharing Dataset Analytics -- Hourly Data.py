# -*- coding: utf-8 -*-
# @Author: Qingqiu Zhang
# @Date:   2022-04-09 18:37:22
# @Last Modified by:   Qingqiu Zhang
# @Last Modified time: 2022-04-09 18:43:59
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from dmba import adjusted_r2_score

hour = pd.read_csv("hour.csv")
hour.head()

result = hour[["hr", "casual", "registered", "cnt"]].groupby(["hr"]).mean()
result = (
    result.stack()
    .reset_index()
    .set_index("hr")
    .rename(columns={"level_1": "cat", 0: "people per hour"})
)
f, axes = plt.subplots(2, sharey=False, figsize=(25, 12))
sns.barplot(x=result.index, y="people per hour", hue="cat", data=result, ax=axes[0])


result = hour[["casual", "registered", "cnt"]].set_index(hour["hr"])
result = (
    result.stack()
    .reset_index()
    .set_index("hr")
    .rename(columns={"level_1": "cat", 0: "people"})
)
sns.violinplot(x=result.index, y="people", hue="cat", data=result, cut=0, ax=axes[1])

cat       = ["season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
hour[cat] = hour[cat].apply(lambda x: x.astype("category"))
dummies   = pd.get_dummies(hour[cat], drop_first=True)
print(dummies.shape)

conti_predictors = ["temp", "atemp", "hum", "windspeed"]
print(hour[conti_predictors].shape)

X = pd.concat([hour[conti_predictors], dummies], axis=1)
X.head()
y = hour.iloc[:, 14:]
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
sc = StandardScaler()
X_train[conti_predictors] = sc.fit_transform(X_train[conti_predictors])
X_test[conti_predictors] = sc.transform(X_test[conti_predictors])

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_pred = pd.DataFrame(y_pred, columns=["casual", "registered", "cnt"]).astype(int)
print(y_pred)
print(adjusted_r2_score(y_test, y_pred, lr))

# Because number of people cannot be negative, we change some data so that the result make sense.
for i in range(y_pred.shape[0]):
    if y_pred["casual"].iloc[i] < 0:
        y_pred["casual"].iloc[i] = 0
    if y_pred["registered"].iloc[i] < 0:
        y_pred["registered"].iloc[i] = 0
        y_pred["cnt"].iloc[i] = y_pred["casual"].iloc[i] + y_pred["registered"].iloc[i]
print(y_pred)
print(adjusted_r2_score(y_test, y_pred, lr))
