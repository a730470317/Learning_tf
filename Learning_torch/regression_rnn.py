from torch import *
import torch
import torch.nn
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

# This file is the demonstration of using RNN to regression y = cos(sin(x))

class Rnn_net(torch.nn.Module):
    def __init__(self):
        super(Rnn_net, self).__init__()
        hidden_size = 100
        input_size = 1
        output_size = 1

        # self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # self.i2o = nn.Linear(input_size + hidden_size, output_size)

        self.rnn = torch.nn.RNN(
            input_size = input_size,
            hidden_size= hidden_size,
            num_layers= 10,
            batch_first= True,
            )

        self.out = torch.nn.Linear(hidden_size,output_size)

    def forward(self, x, hidden_state):
        # combined = torch.cat((input, hidden), 1)
        # hidden_state = self.i2h(combined)
        # rnn_output = self.i2o(combined)

        # print("Run forward")
        rnn_output, hidden_state = self.rnn(x, hidden_state)

        outs = []
        for time_step in range(rnn_output.size(1)):
            outs.append( self.out(rnn_output[:, time_step,:]))
        return torch.stack(outs, dim = 1), hidden_state

if __name__ == "__main__":

    Learning_rate = 0.1

    steps = np.linspace(0 , np.pi*2, 1000, dtype = np.float32)
    # x_np = np.sin(steps)
    # x_np = np.mod(steps, 2*np.pi)
    x_np = np.sin(steps+np.pi)
    y_np = np.cos(steps)

    plt.figure()
    plt.plot(steps, y_np, 'r-', label='target (cos)')
    plt.plot(steps, x_np, 'b-', label='input (cos)')
    plt.legend(loc = 'best')
    plt.ion()

    # rnn_net = Rnn_net().cuda()
    rnn_net = torch.nn.Sequential(
        torch.nn.LSTM(input_size= 1,
                     hidden_size=100,
                     num_layers=10,
                     batch_first=True,
                     ),
        torch.nn.Linear(100, 1)
    ).cuda()
    print(rnn_net)

    # optimizer = torch.optim.Adam(rnn_net.parameters(), lr = Learning_rate)
    optimizer = torch.optim.SGD(rnn_net.parameters(), lr = Learning_rate)
    loss_func = torch.nn.MSELoss()


    x_torch = torch.from_numpy(x_np[np.newaxis, :, np.newaxis]).float().cuda() # RNN data should be 3 dimension
    y_torch = torch.from_numpy(y_np[np.newaxis, :, np.newaxis]).float().cuda()


    print(x_torch.shape)
    print(y_torch.shape)

    h_state = None  # For initial hidden state (all zeros)
    max_time = 100

    rnn_net = torch.load("rnn_net.pkl")
    for t in range(max_time):
        prediction, h_state = rnn_net(x_torch, h_state)
        h_state = h_state.data
        # print(h_state.size())
        loss = loss_func(prediction, y_torch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("time = ", t , " loss = ", loss)
        if(t % (max_time /100) == 0):
            # Learning_rate = Learning_rate*0.90
            # optimizer = torch.optim.Adam(rnn_net.parameters(), lr=Learning_rate)
            optimizer = torch.optim.SGD(rnn_net.parameters(), lr=Learning_rate)

            plt.clf()
            plt.plot(y_np.flatten(), 'r-', label='target')
            plt.plot(prediction.data.cpu().numpy().flatten(), 'b-', label='predict')
            plt.draw()
            plt.pause(0.1)

    torch.save(rnn_net, "rnn_net.pkl")

    plt.pause(0)

