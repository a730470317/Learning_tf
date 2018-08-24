import torch
import gym
import numpy as np
import time

class Policy_net(torch.nn.Module):
    def __init__(self):
        super(Policy_net, self).__init__()
        self.net_1 = torch.nn.Linear(4, 10)
        self.net_2 = torch.nn.Linear(10, 10)
        self.net_out = torch.nn.Linear(10, 2)

    def forward(self, x):
        temp = self.net_1(x)
        temp = self.net_2(temp)
        output = self.net_out(temp)
        output = torch.nn.functional.softmax(output, dim=1)
        return output

def chose_action(action_val):
    if(action_val[0] > action_val[1]):
        return  1
    else:
        return  0

if __name__ == "__main__":
    Learning_rate = 0.2
    Gamma = 0.98
    env = gym.make("CartPole-v0")

    policy_net = Policy_net()
    optimizer = torch.optim.SGD(policy_net.parameters(), lr=Learning_rate)
    print(policy_net)
    for times in range(100000):
        state = env.reset()
        print("----- start %d -----"%(times))
        award_accumulate = 0
        log_state = []
        log_action = []
        log_reward = []
        # Run policy
        while(1):
            # env.render()
            np_state = np.array(state)
            torch_state = torch.from_numpy(np_state[np.newaxis, :]).float()
            action_out = policy_net(torch_state).data.numpy()[0,:]
            action = chose_action(action_out)
            # print(action_out)
            state, reward, done, _ = env.step( action )
            award_accumulate = award_accumulate + reward
            log_state.append(state)
            log_action.append(action)
            log_reward.append(reward)
            # action_out = torch.distributions.Bernoulli(action_out).sample()
            if(done == True):
                break
            # print(np.array(state), reward)

        log_size = len(log_state)
        R_state = log_reward
        for step in range(log_size-1):
            idx=  log_size - 2 - step
            R_state[idx] = log_reward[idx] + Gamma*R_state[idx+1]
        np.set_printoptions(precision=2)
        # print(np.array(R_state))
        # Update policy
        print("Sum of reward is = " , award_accumulate )
        for step in range(log_size):
            net_output = policy_net(torch_state)[0, :]
            # print(net_output)
            if chose_action(net_output.data.numpy()):
                # loss = torch.nn.MSELoss(net_output, torch.tensor([0,1]))
                loss = -torch.dot(net_output, torch.tensor([1.0, 0.0]))
                # loss = - torch.log(idx.float()) / log_size * award_accumulate
            else:
                # loss = torch.nn.MSELoss(net_output, torch.tensor([0, 1]))
                # loss = -torch.dot(net_output, torch.tensor([0.0, 1.0])) / log_size * award_accumulate
                loss = -torch.dot(net_output, torch.tensor([0.0, 1.0]))
            loss = loss* R_state[step]
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    env.close()
    print("finish")
    try:
        exit(0)
    except Exception as e:
        passs