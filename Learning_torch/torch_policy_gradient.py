import torch
import gym
import numpy as np
import time

class Policy_net(torch.nn.Module):
    def __init__(self):
        hidden_size = 10
        super(Policy_net, self).__init__()
        self.net_1 = torch.nn.Linear(4, hidden_size)
        self.net_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.net_out = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        temp = self.net_1(x)
        temp = self.net_2(temp)
        output = self.net_out(temp)
        output = torch.nn.functional.sigmoid(output)
        # output = torch.nn.functional.softmax(output, dim=1)
        return output


if __name__ == "__main__":
    Learning_rate = 0.01
    Gamma = 0.99
    env = gym.make("CartPole-v0")
    env._max_episode_steps = 500
    policy_net = Policy_net()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=Learning_rate)
    print(policy_net)
    log_state = []
    log_action = []
    log_reward = []
    for times in range(100000):
        state = env.reset()
        # print("----- start %d -----"%(times))
        award_accumulate = 0
        # Run policy
        while(1):
            np_state = np.array(state)
            torch_state = torch.from_numpy(np_state[np.newaxis, :]).float()
            net_out = policy_net(torch_state)[0,:]
            action = torch.distributions.Bernoulli(net_out).sample()
            action = action.data.numpy().astype(int)[0]
            # print(action)
            state, reward, done, _ = env.step( action )
            # env.render()

            award_accumulate = award_accumulate + reward
            if (done == True):
                reward = 0
            log_state.append(torch_state)
            log_action.append(action)
            log_reward.append(reward)
            # action_out = torch.distributions.Bernoulli(action_out).sample()
            if(done == True):
                break
            # print(np.array(state), reward)
        print(times," , sum of reward is = ", award_accumulate)
        if(times%5 == 0):
            print("----- update -----")
            log_size = len(log_state)
            R_state = log_reward
            sum_add = 0
            for step in reversed( range(log_size) ):
                idx = step
                if(log_reward[idx] ==0):
                    sum_add = 0
                else:
                    sum_add = Gamma*sum_add + log_reward[idx]
                R_state[idx] = sum_add

            # print(np.array(R_state))

            np.set_printoptions(precision=2)
            reward_mean = np.mean(R_state)
            reward_std = np.std(R_state)
            for step in range(log_size):
                R_state[step] = (R_state[step] - reward_mean)/reward_std

            # print(np.array(R_state))

            optimizer.zero_grad()
            # Update policy

            for step in range(log_size):
                pass
                net_out = policy_net(log_state[step])
                action = torch.autograd.Variable(torch.FloatTensor([log_action[step]]))

                m = torch.distributions.Bernoulli(net_out)
                # print(net_out, ' ---- ', action)
                # print(m.log_prob(action))

                loss = -m.log_prob(action) * R_state[step]
                loss.backward()

            optimizer.step()

            log_state = []
            log_action = []
            log_reward = []

    env.close()
    print("finish")
    try:
        exit(0)
    except Exception as e:
        passs