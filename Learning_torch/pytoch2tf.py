import tensorflow as tf
import torch as torch
import numpy as np
import pickle as pkl
import cv2
import sys, os
import json
import copy

work_dir = "G:/My_research/Airsim/query_data"
# sys.path.append("%s"%work_dir)
sys.path.append("%s/query_data" % work_dir)

from Rapid_trajectory_generator import Rapid_trajectory_generator

deep_drone_path = ("%s/deep_drone/" % work_dir)
trajectory_network_name = "%s/rapid_traj_network_8555200.pkl" % deep_drone_path


def pytorch_net_to_tf(net):
    input_size = net._modules['0'].in_features
    output_size = net._modules[str(len(net._modules) - 1)].out_features
    print("net in size  =  ", input_size)
    print("net out size =  ", output_size)
    tf_lay_w = []
    tf_lay_b = []
    tf_lay = []
    with tf.name_scope("net"):
        net_input = tf.placeholder(tf.float32, [None, input_size], name='net_input');
        try:
            tf_lay.append(net_input)
            for module in net._modules:

                pytoch_lay = net._modules[module]
                if ("Linear" in str(pytoch_lay)):
                    with tf.name_scope("Lay_%s" % len(tf_lay_w)):
                        w = pytoch_lay.weight
                        b = pytoch_lay.bias
                        tf_lay_w.append(tf.convert_to_tensor(copy.deepcopy(w.cpu().data.numpy().T), name="layer_%d_w" % (int(module))))
                        tf_lay_b.append(tf.convert_to_tensor(copy.deepcopy(b.cpu().data.numpy().T), name="layer_%d_b" % (int(module))))
                        # current_wb_idx = len(tf_lay_b) - 1
                        tf_lay.append(tf.matmul(tf_lay[-1], tf_lay_w[-1]) + tf_lay_b[-1])

                        print("[Linear]: w shape = ", w.shape, " b shape = ", b.shape)
                        # break
                else:
                    tf_lay.append(tf.nn.relu(tf_lay[-1], name="relu_%d" % len(tf_lay)))
                    # last_tf_lay = tf.nn.relu(last_tf_lay)
                    # print("[ Relu ]")
                # print(str(net._modules[module]))
        except Exception as e:
            print(e)
    # sess.run(tf.global_variables_initializer())
    return tf_lay


if __name__ == "__main__":

    torch_policy_net = torch.load(trajectory_network_name)
    net = torch_policy_net
    # print(torch_policy_net)
    print("----- Print net -----")

    tf_lay = pytorch_net_to_tf(net)
    sess = tf.InteractiveSession()
    tf.summary.FileWriter("log/", sess.graph)

    # exit(0)

    print(deep_drone_path)
    print(trajectory_network_name)
    json_config = json.load(open("%s/config/config_rapid_trajectory.json" % deep_drone_path, 'r'))
    print(json_config)
    rapid_trajectory_validate = Rapid_trajectory_generator()
    validation_input_data, validation_output_data = rapid_trajectory_validate.load_from_file("%s/traj_0.pkl" % json_config["data_load_dir"])

    tf_out_data = sess.run(tf_lay[-1], feed_dict={tf_lay[0]: validation_input_data})
    print("Test error = ", np.mean(tf_out_data - validation_output_data))

    prediction = torch_policy_net(torch.from_numpy(validation_input_data).float().cuda())
    print(prediction.shape)
    print("Test error = ", np.mean(prediction.data.cpu().numpy() - validation_output_data))
    rapid_trajectory_validate.plot_data("test")
    # rapid_trajectory_validate.output_data = prediction.data.cpu().numpy()
    rapid_trajectory_validate.output_data = tf_out_data
    rapid_trajectory_validate.plot_data_dash("test")
    plt.show()
