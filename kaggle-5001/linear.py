import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv("train.csv")
df["time"] = df["date"].apply(lambda x : x[-5:-3])

df["day"] = df["date"].apply(lambda x : x.split('/')[0])
df["month"] = df["date"].apply(lambda x : x.split('/')[1])
df["year"] = df["date"].apply(lambda x : x.split('/')[2][0:4])
df.drop(["date"], axis = 1, inplace=True)
df.drop(["id"], axis = 1, inplace=True)
ydata = df["speed"]
xdata = pd.get_dummies(df[["time", "day", "month", "year"]])

dft = pd.read_csv("test.csv")
dft["time"] = dft["date"].apply(lambda x : x[-5:-3])
dft["day"] = dft["date"].apply(lambda x : x.split('/')[0])
dft["month"] = dft["date"].apply(lambda x : x.split('/')[1])
dft["year"] = dft["date"].apply(lambda x : x.split('/')[2][0:4])
dft.drop(["date"], axis = 1, inplace=True)
dft.drop(["id"], axis = 1, inplace=True)
dft = pd.get_dummies(dft[["time", "day", "month", "year"]])
temp = dft["year_2018"]
del dft["year_2018"]
dft["year_2017"] = 0
xtest = pd.concat([dft, temp], axis = 1)


x_train, x_test, y_train, y_test = train_test_split(
    xdata, ydata, test_size=0.2, random_state=1)

#############################################################model
linreg = LinearRegression()
linreg.fit(x_train, y_train)

y_pred_train = linreg.predict(x_train)
y_pred = linreg.predict(x_test)

print("trainMSE:",metrics.mean_squared_error(y_train, y_pred_train))
print("testMSE:",metrics.mean_squared_error(y_test, y_pred))

# res = linreg.predict(xtest)
# sub = pd.read_csv("sampleSubmission.csv")
# del sub['speed']
# sub['speed'] = res
# sub.to_csv("result4.csv", index=False)