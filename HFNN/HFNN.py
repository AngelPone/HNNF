from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import random


def data_shuffle(x, y):
    index = list(range(x[0].shape[0]))
    random.shuffle(index)
    x = [i[index] for i in x]
    y = y[index]
    return x, y


class HNNF(nn.Module):

    def __init__(self, nodes):
        super(HNNF, self).__init__()
        self.nodes = nodes
        self.networks = []
        for level in self.nodes:
            self.networks.append([])
            for node in level:
                ff_layer = nn.Linear(node, 1)
                self.networks[-1].append(ff_layer)
        self.fc_final1 = nn.Linear(len(self.nodes)+1, sum(self.nodes[0]))
        self.fc_final2 = nn.Linear(sum(self.nodes[0]), sum(self.nodes[0]))

    def forward(self, x):
        outputs = x[0]
        batch_size = outputs.shape[0]
        level_number = len(self.networks) + 1
        for i in range(level_number-1):
            res = []
            for j in range(len(self.networks[i])):
                start_index = sum(self.nodes[i][0:j])
                end_index = sum(self.nodes[i][0:(j+1)])
                inputs = outputs[:, start_index: end_index]
                res.append(F.relu(self.networks[i][j](inputs)))
            outputs = torch.cat(res, dim=1)
            outputs = torch.cat([outputs, x[i+1]], dim=0)

        outputs = self.fc_final1(torch.cat([outputs[i*batch_size:(i+1)*batch_size]
                                           for i in range(level_number)], dim=1))
        outputs = self.fc_final2(outputs)
        outputs = torch.abs(outputs)
        return outputs

    def loss(self, x, y):
        ouput = self.forward(x)
        return nn.MSELoss(reduction='mean')(ouput, y)

    def fit(self, x, y, test_x, test_y,
            learning_rate=0.01, batch_size=35, epochs=10,
            shuffle=True):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        for t in range(epochs):
            if shuffle:
                x, y = data_shuffle(x, y)
            index = 0
            while True:
                train_x = [i[index*batch_size:(index+1)*batch_size] for i in x]
                train_y = y[index*batch_size:(index+1)*batch_size]
                if len(train_y) == 0:
                    break
                optimizer.zero_grad()
                loss = self.loss(train_x, train_y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                index += 1
                # print(f"epoch {t+1}, {index}/: train loss:{loss.data}")
            test_loss = self.loss(test_x, test_y)
            if t % 5 == 0:
                print(f"epoch {t+1}: test loss: {test_loss.data}")


if __name__ == '__main__':
    nodes = [[2, 2, 1, 4, 4, 1, 3, 1, 3, 6, 8, 3, 4, 3, 2, 3,3,4,2,3,1,1,1,2,2,3,4],
             [6, 5, 4, 4, 3, 3, 2], [7]]

    import pandas as pd
    import numpy as np
    # train data
    train = pd.read_csv("../train.csv")
    train = train.rename({'Unnamed: 0': 'index'}, axis=1)
    train.head()

    # actual values
    tourism = pd.read_csv('../data/TourismData_v4.csv', header=3)
    tourism = tourism.iloc[2:, 3:].reset_index(drop=True).astype('float32')
    tourism.head()

    # prepare test data
    test = pd.read_csv('../arima_forecast.csv')
    test = test.rename({'Unnamed: 0': 'index'}, axis=1)
    test.head()

    models = []
    v = '101'
    for i in range(12):
        h = train[train['index'].map(lambda x: x.startswith(f'h{i + 1}_'))]
        x = []
        for j in range(3, 0, -1):
            x.append(torch.Tensor(h.loc[:, h.columns.map(lambda x: len(x) == j)].values))
        x.append(torch.Tensor(np.expand_dims(h['Total'].values, axis=1)))
        y = torch.Tensor(tourism.iloc[(60 + i):(60 + i + 108)].values)
        model = HNNF(nodes)
        models.append(model)
        print(f'model h={i + 1}')

        h_test = test[test['index'].map(lambda x: x.startswith(f'h{i + 1}_'))]
        x_test = []
        for j in range(3, 0, -1):
            x_test.append(torch.Tensor(h_test.loc[:, h_test.columns.map(lambda x: len(x) == j)].values))
        x_test.append(torch.Tensor(np.expand_dims(h_test['Total'].values, axis=1)))
        y_test = torch.Tensor(tourism.iloc[(168 + i):(168 + i + 61)].values)

        model.fit(x, y, x_test, y_test, batch_size=32, learning_rate=0.5, epochs=50)

        pd.DataFrame(model(x_test).data.numpy()).to_csv(f'forecast_h_{i+1}_v_{v}.csv')
























