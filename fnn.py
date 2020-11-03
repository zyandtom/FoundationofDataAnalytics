if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn import metrics

    df = pd.read_csv("train.csv")
    df["time"] = df["date"].apply(lambda x : int(x[-5:-3]))

    df["day"] = df["date"].apply(lambda x : int(x.split('/')[0]))
    df["month"] = df["date"].apply(lambda x : int(x.split('/')[1]))
    df["year"] = df["date"].apply(lambda x : int(x.split('/')[2][0:4]))
    df.drop(["date"], axis = 1, inplace=True)
    df.drop(["id"], axis = 1, inplace=True)
    ydata = df["speed"]
    #xdata = pd.get_dummies(df[["time", "day", "month", "year"]])
    xdata = df[["time", "day", "month", "year"]]



    dft = pd.read_csv("test.csv")
    dft["time"] = dft["date"].apply(lambda x : int(x[-5:-3]))
    dft["day"] = dft["date"].apply(lambda x : int(x.split('/')[0]))
    dft["month"] = dft["date"].apply(lambda x : int(x.split('/')[1]))
    dft["year"] = dft["date"].apply(lambda x : int(x.split('/')[2][0:4]))
    dft.drop(["date"], axis = 1, inplace=True)
    dft.drop(["id"], axis = 1, inplace=True)
    # dft = pd.get_dummies(dft[["time", "day", "month", "year"]])
    # temp = dft["year_2018"]
    # del dft["year_2018"]
    # dft["year_2017"] = 0
    # dft = pd.concat([dft, temp], axis = 1)

    x_train, x_test, y_train, y_test = train_test_split(
        xdata, ydata, test_size=0.3, random_state=1)



    train = torch.utils.data.TensorDataset(torch.Tensor(np.array(x_train)), torch.Tensor(np.array(y_train)))
    train_loader = torch.utils.data.DataLoader(train, batch_size = 24, shuffle = True, num_workers=2)

    test = torch.utils.data.TensorDataset(torch.Tensor(np.array(x_test)))
    test_loader = torch.utils.data.DataLoader(test, batch_size = 24, shuffle = True, num_workers=2)

    dataiter = iter(train_loader)
    features, out = next(dataiter)

    #########################################################################
    import torch.nn as nn
    import torch.nn.functional as F

    class Mynn(nn.Module):
        def __init__(self):
            super(Mynn, self).__init__()
            #self.bn = nn.BatchNorm1d(69)
            self.fc1 = nn.Linear(69, 100)
            self.fc2 = nn.Linear(100, 1)

        def forward(self, x):
            #x = self.bn(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x
    # class RNN(nn.Module):
    #     def __init__(self):
    #         super(RNN, self).__init__()
    #         self.rnn = nn.LSTM(
    #             input_size=69,
    #             hidden_size=64,
    #             num_layers=1,
    #             batch_first=True
    #         )
    #         self.out = nn.Sequential(
    #             nn.Linear(64, 1)
    #         )
    #
    #     def forward(self, x):
    #         r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
    #         out = self.out(r_out)
    #         return out


    mynn = Mynn()

    ###############################################################
    import torch.optim as optim

    criterion = nn.MSELoss()

    #optimizer = optim.SGD(mynn.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(mynn.parameters(), lr=0.001)
    optimizer = optim.RMSprop(net.parameters(), lr=0.001)

    ###############################################################
    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, truth = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = mynn(inputs)
            outputs = torch.reshape(outputs,(-1,))
            loss = criterion(outputs, truth)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')

    ##############################################################
    mynn.eval()

    res = torch.empty(0)
    with torch.no_grad():
        for data in test_loader:

            data = data[0]
            outputs = mynn(data)
            res = torch.cat((res, outputs))

    print(res)
    ###############################################################
    res = res.numpy()
    print("testMSE:", metrics.mean_squared_error(y_test, res))
    # sub = pd.read_csv("sampleSubmission.csv")
    # del sub['speed']
    # sub['speed'] = res
    # sub.to_csv("result4.csv", index=False)






