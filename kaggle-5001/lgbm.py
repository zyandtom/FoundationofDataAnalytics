import lightgbm as lgb
from lightgbm import plot_importance
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt



df = pd.read_csv("train.csv")
df["time"] = df["date"].apply(lambda x : x[-5:-3])
df["day"] = df["date"].apply(lambda x : x.split('/')[0])
df["month"] = df["date"].apply(lambda x : x.split('/')[1])
df["year"] = df["date"].apply(lambda x : x.split('/')[2][0:4])

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

# df.drop(["date"], axis = 1, inplace=True)
df.drop(["id"], axis = 1, inplace=True)
ydata = df["speed"]
df["time"] = df["time"].apply(lambda x : int(x))
df["day"] = df["day"].apply(lambda x : int(x))
df["month"] = df["month"].apply(lambda x : int(x))
df["year"] = df["year"].apply(lambda x : int(x))
xdata = pd.concat([df[["time", "day", "month", "year"]], df[['holiday', 'spring', 'summer', 'autumn', 'winter']]], axis = 1)

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

##############################################################testdata
dft = pd.read_csv("test.csv")
dft["time"] = dft["date"].apply(lambda x : x[-5:-3])
dft["day"] = dft["date"].apply(lambda x : x.split('/')[0])
dft["month"] = dft["date"].apply(lambda x : x.split('/')[1])
dft["year"] = dft["date"].apply(lambda x : x.split('/')[2][0:4])
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

dft["time"] = dft["time"].apply(lambda x : int(x))
dft["day"] = dft["day"].apply(lambda x : int(x))
dft["month"] = dft["month"].apply(lambda x : int(x))
dft["year"] = dft["year"].apply(lambda x : int(x))
xtest = dft
##########################################################################weekend
weekendt = [1]
daytindex = 1
for i in range(1, len(dayt)):
    if int(dayt[i-1]) == int(dayt[i]):
        weekendt.append(weekendt[-1])
    else:
        daytindex += 1
        if (daytindex%6 == 1 or daytindex%6 == 2) and daytindex > 2:
            weekendt.append(1)
        else:
            weekendt.append(0)
xtest["weekend"] = weekendt

##################################################################
x_train, x_test, y_train, y_test = train_test_split(
    xdata, ydata, test_size=0.3, random_state=1)

params = {
    'boosting_type': 'gbdt',  # 训练方式
    'objective': 'regression',
    'metric': 'mse',
    'min_data_in_leaf': 5,
}


lgb_train = lgb.Dataset(x_train, label=y_train)
lgb_eval = lgb.Dataset(x_test, label=y_test)
params['learning_rate'] = 0.5
gbm = lgb.train(
    params,  # 参数字典
    lgb_train,  # 训练集
    valid_sets=lgb_eval,  # 验证集
    num_boost_round=1000,  # 迭代次数
    early_stopping_rounds=50  # 早停次数
)


################################################################################
# predict_y_train = gbm.predict(x_train)
predict_y_test = gbm.predict(x_test)

# print("trainMSE:",metrics.mean_squared_error(y_train, predict_y_train))
print("testMSE:",metrics.mean_squared_error(y_test, predict_y_test))

# plot_importance(gbm)
# plt.show()
# res = gbm.predict(xtest)
# sub = pd.read_csv("sampleSubmission.csv")
# del sub['speed']
# sub['speed'] = res
# sub.to_csv("result5.csv", index=False)

plot_importance(gbm)
plt.show()