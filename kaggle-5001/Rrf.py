import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("train.csv")
df["time"] = df["date"].apply(lambda x : x[-5:-3])

df["day"] = df["date"].apply(lambda x : x.split('/')[0])
df["month"] = df["date"].apply(lambda x : x.split('/')[1])
df["year"] = df["date"].apply(lambda x : x.split('/')[2][0:4])
df.drop(["date"], axis = 1, inplace=True)
df.drop(["id"], axis = 1, inplace=True)
ydata = df["speed"]
xdata = pd.get_dummies(df[["time", "day", "month", "year"]])

##############################(weekend)
day = df["day"]
weekend = [1]
dayindex = 1
for i in range(1, len(day)):
    if int(day[i-1]) == int(day[i]):
        weekend.append(weekend[-1])
    else:
        dayindex += 1
        if (dayindex%6 == 1 or dayindex%6 == 2) and dayindex > 2:
            weekend.append(1)
        else:
            weekend.append(0)
xdata["weekend"] = weekend


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

##################################################################
x_train, x_test, y_train, y_test = train_test_split(
    xdata, ydata, test_size=0.3, random_state=1)

# tuned_parameters = [{'n_estimators': [150], 'max_depth': [50]}]
#
#
# model = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=1,
#                    scoring='neg_mean_squared_error')
model = RandomForestRegressor(n_estimators=150, max_depth=70)#150 60
model.fit(x_train, y_train)

#################################################3
predict_y_train = model.predict(x_train)
predict_y_test = model.predict(x_test)

print("trainMSE:",metrics.mean_squared_error(y_train, predict_y_train))
print("testMSE:",metrics.mean_squared_error(y_test, predict_y_test))
# print(model.best_params_)
# print(model.best_score_ )
# print(model.cv_results_['mean_test_score'])

# res = model.predict(xtest)
# sub = pd.read_csv("sampleSubmission.csv")
# del sub['speed']
# sub['speed'] = res
# sub.to_csv("result4.csv", index=False)

