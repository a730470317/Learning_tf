from __future__ import print_function
from torch import *
import torch.nn
import torch.nn.functional
import torch.optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import numpy as np
import cv2 as cv2


# This python is a demonstration  to using network to approximate to " y = a^2 + b^3 "

class Net1(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output, activation_fun='relu'):
        super(Net1, self).__init__()
        self.hidden_0 = torch.nn.Linear(n_features, n_hidden)
        self.hidden_1 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden_2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden_3 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
        self.activation_fun = activation_fun

    def forward(self, x):
        temp = self.hidden_0(x)
        temp = self.hidden_1(temp)
        temp = self.hidden_2(temp)
        temp = self.hidden_3(temp)
        if (self.activation_fun == 'relu'):
            # print("using relu")
            temp = torch.nn.functional.relu(temp)
        elif (self.activation_fun == 'sigmoid'):
            temp = torch.nn.functional.sigmoid(temp)
            # print("using sigmoid")
        elif (self.activation_fun == 'tanh'):
            temp = torch.nn.functional.tanh(temp)
            # print("using tanh")
        else:
            temp = torch.nn.functional.relu(temp)
        temp = self.predict(temp)
        return temp


def visualize(av, bv, y, y_predict, sample_size):
    plt.ion()
    fig = plt.figure("compare")
    ax3d = fig.gca( projection='3d')
    ax3d.cla()

    av = av.reshape(sample_size, sample_size)
    bv = bv.reshape(sample_size, sample_size)
    y = y.reshape(sample_size, sample_size)

    y_predict = net(torch_input_data).cpu().data.numpy()
    print(y_predict.shape)
    y_predict = y_predict.reshape(sample_size, sample_size)
    print(y_predict.shape)
    ax3d.plot_wireframe(av, bv, y, rstride=5, cstride=5, label='Raw function')  # Plot a basic wireframe.
    plt.pause(0.01)

    ax3d.plot_wireframe(av, bv, y_predict, rstride=5, cstride=5, color='b', label='Approximate function')  # Plot a basic wireframe.
    ax3d.legend()
    plt.pause(0.01)

    fig = plt.figure("diff")
    ax3d = fig.gca(projection='3d')
    ax3d.cla()
    ax3d.plot_surface(av, bv, y-y_predict, cmap=cm.coolwarm, linewidth=0, antialiased=False) # Plot a surface
    plt.pause(0.01)


if __name__ == "__main__":
    # Generate testing data
    if_visualize = 0
    torch.manual_seed(1)  # reproducible
    bound = 1
    sample_size = 50
    a = np.linspace(-bound, bound, sample_size)
    b = np.linspace(-bound, bound, sample_size)
    av, bv = np.meshgrid(a, b)
    av = av.ravel()  # expand to one-dimensional array
    bv = bv.ravel()
    size = len(av)

    input_data = np.zeros((size, 2))

    for step, _a in enumerate(zip(av, bv)):
        input_data[step, 0] = _a[0]
        input_data[step, 1] = _a[1]

    y = 2*np.power(av, 2) + np.power(bv, 4) + np.sin(av) + np.cos(bv)
    y = y.ravel()

    torch_input_data = torch.from_numpy(input_data).float().cuda()

    torch_y = torch.from_numpy(y).float().cuda()
    torch_y = torch.unsqueeze(torch_y, dim=1)

    print("y_shape = ", torch_y.shape)
    print("input_data shape = ", torch_input_data.shape)

    n_hidden_per_layer = 100
    # net = Net1(2,100,1,'relu')                      # method 1
    # print(net)
    net = torch.nn.Sequential(
        torch.nn.Linear(2, n_hidden_per_layer),
        torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_per_layer, n_hidden_per_layer),
        # torch.nn.Sigmoid(),
        # torch.nn.Softplus(),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_per_layer, 1) ).cuda()    # method 2
    print(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    # optimizer = torch.optim.Adam(net.parameters(), lr= 0.1,betas=(0.9, 0.99))
    loss_func = torch.nn.MSELoss()

    for t in range(10000):
        prediction = net(torch_input_data)
        loss = loss_func(prediction, torch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t % 100 == 0:
            print('t =', t, " loss", loss)
            if(if_visualize):
                visualize(av, bv, y, prediction, sample_size)
        if (t>1000):
            optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
        if (t > 2000):
            optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        if (t > 3000):
            optimizer = torch.optim.SGD(net.parameters(), lr=0.005)
        if (t > 4000):
            optimizer = torch.optim.SGD(net.parameters(), lr=0.0025)
        if (t > 8000):
            optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    visualize(av, bv, y, prediction, sample_size)
    plt.pause(0)