import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
import datetime as dt
import numpy as np


max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))

weather = pd.read_csv('./wind.csv', parse_dates=['Time'])
weather['temperature'] =  weather[['temperature']].apply(max_min_scaler)
weather['wet'] =  weather[['wet']].apply(max_min_scaler)
weather['cloud'] =  weather[['cloud']].apply(max_min_scaler)
weather['rain'] =  weather[['rain']].apply(max_min_scaler)
weather['wind'] =  weather[['wind']].apply(max_min_scaler)

weather['year'] = weather['Time'].dt.year
weather['month'] = weather['Time'].dt.month
weather['day'] = weather['Time'].dt.day
weather=weather.drop('Time',axis=1)
weather['year'] = 1  #2018
weather['year'].iloc[:365] = 0  #2017

df = pd.read_csv("train.csv")
df["time"] = df["date"].apply(lambda x : int(x[-5:-3]))
df["day"] = df["date"].apply(lambda x : int(x.split('/')[0]))
df["month"] = df["date"].apply(lambda x : int(x.split('/')[1]))
df["year"] = df["date"].apply(lambda x : int(x.split('/')[2][0:4]))

########################################################holiday
df['holiday'] = 0
for i in range(14006):
    if df["month"][i] == "1" and df['year'][i] == '2017' and df["day"][i] in ['1', '2', '27', '28', '29', '30', '31']:
        df['holiday'][i] = 1
    elif df["month"][i] == "2" and df['year'][i] == '2017' and df["day"][i] in ['1', '2']:
        df['holiday'][i] = 1
    elif df["month"][i] == "4" and df['year'][i] == '2017' and df["day"][i] in ['2', '3', '4', '29', '30']:
        df['holiday'][i] = 1
    elif df["month"][i] == "5" and df['year'][i] == '2017' and df["day"][i] in ['1', '28', '29', '30']:
        df['holiday'][i] = 1
    elif df["month"][i] == "10" and df['year'][i] == '2017' and df["day"][i] in ['1', '2', '3', '4', '5', '6', '7', '8']:
        df['holiday'][i] = 1
    elif df["month"][i] == "12" and df['year'][i] == '2017' and df["day"][i] in ['31']:
        df['holiday'][i] = 1
    elif df["month"][i] == "1" and df['year'][i] == '2018' and df["day"][i] in ['1']:
        df['holiday'][i] = 1
    elif df["month"][i] == "2" and df['year'][i] == '2018' and df["day"][i] in ['15', '16', '17', '18', '19', '20', '21']:
        df['holiday'][i] = 1
    elif df["month"][i] == "4" and df['year'][i] == '2018' and df["day"][i] in ['5', '6', '7']:
        df['holiday'][i] = 1
    elif df["month"][i] == "6" and df['year'][i] == '2018' and df["day"][i] in ['16', '17', '18']:
        df['holiday'][i] = 1
    elif df["month"][i] == "9" and df['year'][i] == '2018' and df["day"][i] in ['24']:
        df['holiday'][i] = 1
    elif df["month"][i] == "10" and df['year'][i] == '2018' and df["day"][i] in ['1', '2', '3', '4', '5', '6', '7']:
        df['holiday'][i] = 1

###########################################################################seasons
df['spring'] = 0
df['summer'] = 0
df['autumn'] = 0
df['winter'] = 0
df['winter'].iloc[0:783] = 1
df['spring'].iloc[783:2967] = 1
df['summer'].iloc[2967:5222] = 1
df['autumn'].iloc[5222:7430] = 1
df['winter'].iloc[7430:9246] = 1
df['spring'].iloc[9246:10566] = 1
df['summer'].iloc[10566:11975] = 1
df['autumn'].iloc[11975:13260] = 1
df['winter'].iloc[13260:14006] = 1
######################################################################################

df.drop(["date"], axis = 1, inplace=True)
df.drop(["id"], axis = 1, inplace=True)
ydata = df["speed"]
xdata = df[["time", "day", "month", 'spring', 'summer', 'autumn', 'winter']]
xdata['year'] = 0
xdata['year'].iloc[8750:14006] = 1
##############################(weekend)
day = df["day"]
weekend = [7]
dayindex = 1
for i in range(1, len(day)):
    if int(day[i-1]) == int(day[i]):
        weekend.append(weekend[-1])
    else:
        dayindex += 1
        if dayindex%7 == 2:
            weekend.append(1)
        elif dayindex%7 == 3:
            weekend.append(2)
        elif dayindex%7 == 4:
            weekend.append(3)
        elif dayindex%7 == 5:
            weekend.append(4)
        elif dayindex%7 == 6:
            weekend.append(5)
        elif dayindex%7 == 0:
            weekend.append(6)
        else: weekend.append(7)
xdata["weekend"] = weekend

##############################################################testdata
dft = pd.read_csv("test.csv")
dft["time"] = dft["date"].apply(lambda x : int(x[-5:-3]))
dft["day"] = dft["date"].apply(lambda x : int(x.split('/')[0]))
dft["month"] = dft["date"].apply(lambda x : int(x.split('/')[1]))
dft["year"] = dft["date"].apply(lambda x : int(x.split('/')[2][0:4]))
dayt = dft["day"]

#################################################################holiday
dft['holiday'] = 0
for i in range(len(dft)):
    if dft["month"][i] == "1" and dft['year'][i] == '2018' and dft["day"][i] in ['1']:
        dft['holiday'][i] = 1
    elif dft["month"][i] == "2" and dft['year'][i] == '2018' and dft["day"][i] in ['15', '16', '17', '18', '19', '20', '21']:
        dft['holiday'][i] = 1
    elif dft["month"][i] == "4" and dft['year'][i] == '2018' and dft["day"][i] in ['5', '6', '7']:
        dft['holiday'][i] = 1
    elif dft["month"][i] == "6" and dft['year'][i] == '2018' and dft["day"][i] in ['16', '17', '18']:
        dft['holiday'][i] = 1
    elif dft["month"][i] == "9" and dft['year'][i] == '2018' and dft["day"][i] in ['24']:
        dft['holiday'][i] = 1
    elif dft["month"][i] == "10" and dft['year'][i] == '2018' and dft["day"][i] in ['1', '2', '3', '4', '5', '6', '7']:
        dft['holiday'][i] = 1

#####################################################################seasons
dft['spring'] = 0
dft['summer'] = 0
dft['autumn'] = 0
dft['winter'] = 0
dft['winter'].iloc[0:320] = 1
dft['spring'].iloc[320:1160] = 1
dft['summer'].iloc[1160:2007] = 1
dft['autumn'].iloc[2007:2930] = 1
dft['winter'].iloc[2930:3504] = 1

###############################################################################

dft.drop(["date"], axis = 1, inplace=True)
dft.drop(["id"], axis = 1, inplace=True)

# tempdf = pd.get_dummies(dft[["time", "day", "month"]])
# temp = tempdf["year_2018"]
# del tempdf["year_2018"]
# tempdf["year_2017"] = 0
# tempdf = pd.concat([tempdf, temp], axis = 1)

xtest = dft[["time", "day", "month", 'spring', 'summer', 'autumn', 'winter']]
xtest['year'] = 1

##########################################################################weekend
weekendt = [1]
daytindex = 1
for i in range(1, len(dayt)):
    if int(dayt[i-1]) == int(dayt[i]):
        weekendt.append(weekendt[-1])
    else:
        daytindex += 1
        if daytindex%7 == 1:
            weekendt.append(1)
        elif daytindex%7 == 2:
            weekendt.append(2)
        elif daytindex%7 == 3:
            weekendt.append(3)
        elif daytindex%7 == 4:
            weekendt.append(4)
        elif daytindex%7 == 5:
            weekendt.append(5)
        elif daytindex%7 == 6:
            weekendt.append(6)
        else: weekendt.append(7)
xtest["weekend"] = weekendt

############################################
xdata = pd.merge(xdata, weather, on = ['year','month', 'day'], how = 'left')
xtest = pd.merge(xtest, weather, on = ['year','month', 'day'], how = 'left')
##################################################################
x_train, x_test, y_train, y_test = train_test_split(
    xdata, ydata, test_size=0.2, random_state=1)

params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'gamma': 0.05,
    'max_depth': 50,
    'lambda': 3,
    'subsample': 0.4,
    'colsample_bytree': 1,
    'min_child_weight': 0,
    'eta': 0.01,
    'seed': 0,
    'nthread': 4
}
dtest = xgb.DMatrix(x_test, y_test)
dtrain = xgb.DMatrix(x_train, y_train)
num_rounds = 1030
plst = list(params.items())

watchlist = [(dtrain,'train'),(dtest,'val')]
model = xgb.train(plst, dtrain, num_rounds, evals=watchlist)

################################################################################
# predict_y_train = model.predict(xgb.DMatrix(x_train))
# predict_y_test = model.predict(xgb.DMatrix(x_test))
#
# print("trainMSE:",metrics.mean_squared_error(y_train, predict_y_train))
# print("testMSE:",metrics.mean_squared_error(y_test, predict_y_test))
#
#
# plot_importance(model)
# plt.show()



dtrain = xgb.DMatrix(xdata, ydata)
num_rounds = 1200
plst = list(params.items())
model = xgb.train(plst, dtrain, num_rounds)

res = model.predict(xgb.DMatrix(xtest))
sub = pd.read_csv("sampleSubmission.csv")
del sub['speed']
sub['speed'] = res
sub.to_csv("result.csv", index=False)