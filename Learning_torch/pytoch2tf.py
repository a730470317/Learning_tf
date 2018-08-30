import tensorflow as tf
import torch as torch
import torch.utils.data as Data
import numpy as np
import pickle as pkl
import cv2
import sys, os
import json
import copy
import matplotlib.pyplot as plt
work_dir = "G:/My_research/Airsim/query_data"
# sys.path.append("%s"%work_dir)
sys.path.append("%s/query_data" % work_dir)

from Rapid_trajectory_generator import Rapid_trajectory_generator

deep_drone_path = ("%s/deep_drone/" % work_dir)
trajectory_network_name = "%s/rapid_traj_network_8555200.pkl" % deep_drone_path

class Plannning_net_tf:
    def __init__(self):
        self.tf_lay = []
        self.tf_w = []
        self.tf_b = []
        self.train = None
        self.net_input = None
        self.net_output = None
        self.target_output = None  # For supervised trainning
        self.weighted_val = [4, 2, 1]
        self.learn_rate = 0.000001 * 0.5

    def train_loop(self):
        print(self.net_output, self.target_output)
        diff = self.net_output - self.target_output
        pos_err = tf.slice(diff, [0,0], [-1, 3])
        spd_err = tf.slice(diff, [0,3], [-1, 3])
        acc_err = tf.slice(diff, [0,6], [-1, 3])

        # pos_err = self.net_output[:, (0, 1, 2)] - self.target_output[:, (0, 1, 2)]
        # spd_err = self.net_output[:, (3, 4, 5)] - self.target_output[:, (3, 4, 5)]
        # acc_err = self.net_output[:, (6, 7, 8)] - self.target_output[:, (6, 7, 8)]
        # loss = tf.reduce_mean(pos_err * self.weighted_val[0] + spd_err * self.weighted_val[1] + acc_err * self.weighted_val[2], name="loss")
        loss = tf.reduce_mean(tf.norm(pos_err, axis= 1, keepdims= True) * self.weighted_val[0] + \
                              tf.norm(spd_err, axis= 1, keepdims= True) * self.weighted_val[1] + \
                              tf.norm(acc_err, axis= 1, keepdims= True) * self.weighted_val[2] ,
                              name="loss")
        self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(loss)
        pass

    def pytorch_net_to_tf(self, net):
        self.input_size = net._modules['0'].in_features
        self.output_size = net._modules[str(len(net._modules) - 1)].out_features
        print("net in size  =  ", self.input_size)  # in size = 17
        print("net out size =  ", self.output_size) # out size = 9
        self.target_output = tf.placeholder(tf.float32, [None, self.output_size], name='net_input');
        self.tf_lay = []
        self.tf_w = []
        self.tf_b = []

        with tf.name_scope("net"):
            self.net_input = tf.placeholder(tf.float32, [None, self.input_size], name='net_input');

            self.tf_lay.append(self.net_input)
            for module in net._modules:

                pytoch_lay = net._modules[module]
                if ("Linear" in str(pytoch_lay)):
                    with tf.name_scope("Lay_%s" % len(self.tf_w)):
                        w = pytoch_lay.weight
                        b = pytoch_lay.bias
                        self.tf_w.append(tf.Variable( tf.convert_to_tensor(copy.deepcopy(w.cpu().data.numpy().T), name="layer_%d_w" % (int(module)))) )
                        self.tf_b.append(tf.Variable( tf.convert_to_tensor(copy.deepcopy(b.cpu().data.numpy().T), name="layer_%d_b" % (int(module)))) )
                        # current_wb_idx = len(tf_lay_b) - 1
                        self.tf_lay.append(tf.matmul(self.tf_lay[-1], self.tf_w[-1]) + self.tf_b[-1])

                        # print("[Linear]: w shape = ", w.shape, " b shape = ", b.shape)
                        # break
                else:
                    self.tf_lay.append(tf.nn.relu(self.tf_lay[-1], name="relu_%d" % len(self.tf_lay)))
                    # last_tf_lay = tf.nn.relu(last_tf_lay)
                    # print("[ Relu ]")
                # print(str(net._modules[module]))
        self.net_output = self.tf_lay[-1]
        # self.train_loop()
        # sess.run(tf.global_variables_initializer())
        return self.tf_lay


if __name__ == "__main__":

    torch_policy_net = torch.load(trajectory_network_name)
    net = torch_policy_net
    # print(torch_policy_net)
    print("----- Print net -----")
    tf_net = Plannning_net_tf()
    tf_lay = tf_net.pytorch_net_to_tf(net)
    tf_net.train_loop()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter("log/", sess.graph)
    saver = tf.train.Saver()
    saver.save(sess,"./tf_saver.ckpt")
    # exit(0)

    print(deep_drone_path)
    print(trajectory_network_name)
    json_config = json.load(open("%s/config/config_rapid_trajectory.json" % deep_drone_path, 'r'))
    print(json_config)
    rapid_trajectory_validate = Rapid_trajectory_generator()
    validation_input_data, validation_output_data = rapid_trajectory_validate.load_from_file("%s/traj_0.pkl" % json_config["data_load_dir"])

    tf_out_data = sess.run(tf_net.net_output, feed_dict={tf_net.net_input: validation_input_data})
    print("Test error = ", np.mean(tf_out_data - validation_output_data))

    prediction = torch_policy_net(torch.from_numpy(validation_input_data).float().cuda())
    print(prediction.shape)
    print("Test error = ", np.mean(prediction.data.cpu().numpy() - validation_output_data))

    rapid_trajectory_validate.plot_data("test")
    # rapid_trajectory_validate.output_data = prediction.data.cpu().numpy()
    rapid_trajectory_validate.output_data = tf_out_data
    # rapid_trajectory_validate.plot_data_dash("test")
    # plt.show()

    rapid_trajectory = Rapid_trajectory_generator()
    sample_size = 15000
    t_start = cv2.getTickCount()
    input_data, output_data = rapid_trajectory.load_from_file("%s/batch_%d.pkl" % (json_config["data_load_dir"], sample_size))
    torch_input_data = torch.from_numpy(input_data).float()
    torch_output_data = torch.from_numpy(output_data).float()
    BATCH_SIZE = 100*1000
    torch_dataset = Data.TensorDataset(torch_input_data.cpu(), torch_output_data.cpu())
    loader = Data.DataLoader(
            dataset=torch_dataset,  # torch TensorDataset format
            batch_size=BATCH_SIZE,  # mini batch size
            shuffle=True,  # if shuffle the data
            num_workers=7,  # using how many worker.
    )
    print("Load finish.")
    print("Set data finish.")
    for epoch in range(10000):
        for loop_1, (batch_x, batch_y) in enumerate(loader):
            np_data_x = batch_x.data.numpy()
            np_data_y = batch_y.data.numpy()
            loop_2_times = 1000
            for loop_2 in range(loop_2_times):
                tf_net.train_step.run({tf_net.net_input: np_data_x, tf_net.target_output: np_data_y})
                t = epoch * len(loader) + loop_1 * loop_2_times + loop_2 + 0
                if(t%1000==0 and t!=0):
                    saver.save(sess, "./tf_net/tf_saver_%d.ckpt"%t)

                if(t%100==0):
                    error = np.mean(sess.run(tf_net.net_output, feed_dict={tf_net.net_input: np_data_x}) - np_data_y)
                    log_str = "epoch = " + str(epoch) + ' |loop_1 = ' + str(loop_1) + ' |loop_2 = ' + str(loop_2) + ' |t =' + str(t) + " |loss = " + str(error)
                    print(log_str)
                if(t%10000 == 0):
                    fig = plt.figure("test")
                    plt.clf()
                    fig.set_size_inches(16 * 2, 9 * 2)
                    rapid_trajectory_validate.output_data = validation_output_data
                    rapid_trajectory_validate.plot_data("test")
                    tf_out_data = sess.run(tf_net.net_output, feed_dict={tf_net.net_input: validation_input_data})
                    rapid_trajectory_validate.output_data = tf_out_data
                    rapid_trajectory_validate.plot_data_dash("test")
                    fig.savefig("%s/test_%d.png" % ("./tf_net", t))