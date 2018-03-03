import gym
import tensorflow as tf
import numpy as np
# See source file in
class policy_gradient:

    def build_mlp(self, layer_size):
        with tf.name_scope("input"):
            self.ob_input = tf.placeholder(tf.float32, [None, self.m_observation_size], name='ob_input');
            self.ac_input = tf.placeholder(tf.int32, [None, ], name='ac_input');
            self.ac_value = tf.placeholder(tf.float32, [None, ], name='ac_value');
        with tf.name_scope("policy_network"):
            self.fc1 = tf.layers.dense(self.ob_input,
                                       layer_size,
                                       kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                       bias_initializer=tf.constant_initializer(0.1),
                                       name = "fc1");
            self.fc2 = tf.layers.dense(self.fc1,
                                       self.m_action_size,
                                       kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                       bias_initializer=tf.constant_initializer(0.1),
                                       name = "fc2");
            self.all_action = tf.nn.softmax(self.fc2,name = 'all_action');
            self.chose_action = tf.argmax(self.all_action,1)
        with tf.name_scope("loss"):
            pass
            # loss_pre = tf.reduce_mean( -tf.reduce_sum(-self.all_action*tf.log(self.ac_input)))
            # loss_pre = tf.reduce_sum(-tf.cast(self.ac_input,tf.float32) * tf.log(self.all_action),name='loss_pre')
            loss_pre = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.fc2, labels=self.ac_input)
            loss = tf.reduce_mean(loss_pre*self.ac_value,name='loss')
            self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(loss)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        return self.sess;

    def __init__(self,
                 observation_size,
                 action_size,
                 learn_rate = 1.0,
                 reward_decay = 0.99
                 ):
        self.learn_rate = learn_rate
        self.reward_decay = reward_decay
        print("run init function.")
        self.obs_vec = []
        self.rew_vec = []
        self.acs_vec = []
        self.m_observation_size = observation_size
        self.m_action_size = action_size
        sess = self.build_mlp(10);
        tf.summary.FileWriter("log/", sess.graph)

    def deploy(self,observation):
        # print("Run deploy, obs = " ,observation)
        # print(observation[np.newaxis, :])
        if 1:
            prob_weights = self.sess.run(self.all_action, feed_dict={self.ob_input: observation[np.newaxis, :]})
            # print(range(prob_weights.shape[1]),' ' ,prob_weights.ravel())
            act_predict = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
            # print("wright: " ,prob_weights, "res: " , act_predict)
        else:
            act_predict = self.sess.run(self.chose_action, feed_dict={self.ob_input: observation[np.newaxis, :]})[0]
            # print("chose action: ", act_predict)
        return  act_predict;

    def store(self, observation, action,reward ):
        self.obs_vec.append(observation)
        self.acs_vec.append(action)
        self.rew_vec.append(reward)

    def compute_value(self):
        self.gamma = self.reward_decay
        discounted_ep_rs = np.zeros_like(self.rew_vec)
        running_add = 0
        for t in reversed(range(0, len(self.rew_vec))):
            running_add = running_add * self.gamma + self.rew_vec[t]
            discounted_ep_rs[t] = running_add
        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def train(self):
        value_vec = self.compute_value()
        print(np.vstack(self.obs_vec).shape, "  ", np.array(self.acs_vec).shape,"  ", value_vec.shape)
        self.sess.run(self.train_step,feed_dict={self.ob_input : np.vstack(self.obs_vec),
                                                                 self.ac_input : np.array(self.acs_vec),
                                                                 self.ac_value : value_vec})
        self.obs_vec = []
        self.rew_vec = []
        self.acs_vec = []

if __name__ == "__main__":
    # print(tf.__version__)
    env = gym.make('CartPole-v0') #Src file :/home/zivlin/App/gym/gym/envs/classic_control/cartpole.py:
    env.reset()
    env = env.unwrapped # This is very important, else your env will run 200 times.
    print(env.observation_space.shape, env.action_space.n)

    observation, reward, done, info = env.step(env.action_space.sample())  # take a random action
    pg = policy_gradient(observation_size =  env.observation_space.shape[0],
                         action_size = env.action_space.n,
                         learn_rate=0.05,
                         reward_decay=0.99)
    loop_i =0;
    sum_reward = 0
    run_time = 0
    while (loop_i < 10000):
        if(sum_reward > 1000 ):
            env.render()
        # observation, reward, done, info = env.step(env.action_space.sample())  # take a random action
        action_take = pg.deploy(observation);
        observation, reward, done, info = env.step(action_take)  # take a random action
        # print([observation, reward, done, info])
        if(done):
            print("Observation: ",observation);
            sum_reward = sum(pg.rew_vec)
            print("loop: " ,loop_i ,"rum_time: ",run_time, " sum reward is: ", sum_reward )
            pg.train();
            env.reset()
            run_time = 0
            loop_i = loop_i +1
            # break
        else:
            # sum_reward = sum_reward +reward
            run_time = run_time+1
            pg.store(observation, action_take, reward)
    print("finish")
