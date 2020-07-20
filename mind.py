import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

import random
import numpy as np

class Mind:
    BATCH_SIZE = 256
    GAMMA = 0.98
    EPS_START = 0.9999
    EPS_END = 0
    EPS_DECAY = 100000
    TAU = 0.05
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, input_size, num_actions, lock, queue, destination = None, memory_length=1000000):
        self.network = DQN(input_size, num_actions)
        self.target_network = DQN(input_size, num_actions)
        self.lock = lock
        self.queue = queue
        self.losses = []
        self.network.share_memory()
        self.target_network.share_memory()

        self.input_size, self.num_actions = input_size, num_actions


        self.memory = ReplayMemory(memory_length)
        self.optimizer = optim.Adam(self.network.parameters(), 0.001)
        self.steps_done = 0
        self.num_actions = num_actions

        self.target_network.load_state_dict(self.network.state_dict())
        self.input_size = input_size
        self.num_cpu = mp.cpu_count() // 2

    def save(self, name, type):
        #torch.save(self.network.state_dict(), "%s/%s_network.pth" % (name, type))
        #torch.save(self.target_network.state_dict(), "%s/%s_target_network.pth" % (name, type))
        #torch.save(self.optimizer.state_dict(), "%s/%s_optimizer.pth" % (name, type))

        #states, ages, actions, next_states, rewards, dones = zip(*self.memory.memory)

        """
        np.save("%s/%s_states.npy" % (name, type), states)
        np.save("%s/%s_ages.npy" % (name, type), ages)
        np.save("%s/%s_actions.npy" % (name, type), actions)
        np.save("%s/%s_next_states.npy" % (name, type), next_states)
        np.save("%s/%s_rewards.npy" % (name, type), rewards)
        np.save("%s/%s_dones.npy" % (name, type), dones)

        np.save("%s/%s_memory_pos.npy" % (name, type), np.array([self.memory.position]))
        """
        #np.save("%s/%s_loss.npy" % (name, type), np.array(self.losses))
        pass

    def load(self, name, type, iter):
        """
        self.network.load_state_dict(torch.load("%s/%s_network.pth" % (name, type)))
        self.target_network.load_state_dict(torch.load("%s/%s_target_network.pth" % (name, type)))
        self.optimizer.load_state_dict(torch.load("%s/%s_optimizer.pth" % (name, type)))

        self.losses = list(np.load("%s/%s_loss.npy" % (name, type)))
        states = np.load("%s/%s_states.npy" % (name, type))
        ages = np.load("%s/%s_ages.npy" % (name, type))
        actions = np.load("%s/%s_actions.npy" % (name, type))
        next_states = np.load("%s/%s_next_states.npy" % (name, type))
        rewards = np.load("%s/%s_rewards.npy" % (name, type))
        dones = np.load("%s/%s_dones.npy" % (name, type))

        self.memory.memory = list(zip(states, ages, actions, next_states, rewards, dones))

        self.memory.position = int(np.load("%s/%s_memory_pos.npy" % (name, type))[0])
        self.steps_done = iter
        """
        pass
    
    def get_input_size(self):
        return self.input_size

    def get_output_size(self):
        return self.num_actions

    def get_losses(self):
        return self.losses

    def decide(self, state, age, type):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            np.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                state = torch.FloatTensor([[state]], device=self.device)
                age = torch.FloatTensor([[age]], device=self.device)
                q_values = self.network(type * state, age)
                return q_values.max(1)[1].view(1, 1).detach().item()
        else:
            rand = [[random.randrange(self.num_actions)]]
            return torch.tensor(rand, device=self.device, dtype=torch.long).detach().item()

    def remember(self, vals):
        self.memory.push(vals)

    def copy(self):
        net = DQN(self.input_size, self.num_actions)
        target_net = DQN(self.input_size, self.num_actions)
        optimizer = optim.Adam(net.parameters(), 0.001)
        optimizer.load_state_dict(self.optimizer.state_dict())
        net.load_state_dict(self.network.state_dict())
        target_net.load_state_dict(self.target_network.state_dict())

        return net, target_net, optimizer

    def opt(self, data, lock, queue, type):
        batch_state, batch_age, batch_action, batch_next_state, batch_done, expected_q_values = data
        current_q_values = self.network(type * batch_state, batch_age).gather(1, batch_action)
        max_next_q_values = self.target_network(type * batch_next_state, batch_age).detach().max(1)[0]

        for i, done in enumerate(batch_done):
            if not done:
                expected_q_values[i] += (self.GAMMA * max_next_q_values[i])

        loss = F.mse_loss(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        queue.put(loss.item())
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.TAU * param.data + target_param.data * (1.0 - self.TAU))

    def get_data(self):
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch_state, batch_age, batch_action, batch_next_state, batch_reward, batch_done = zip(*transitions)
        batch_state = torch.cat([torch.FloatTensor(s) for s in batch_state])
        batch_age = torch.cat([torch.FloatTensor(s) for s in batch_age]).view((self.BATCH_SIZE, 1))
        batch_action = torch.cat([torch.LongTensor(s) for s in batch_action]).view((self.BATCH_SIZE, 1))
        batch_reward = torch.cat([torch.FloatTensor(s) for s in batch_reward])
        batch_next_state = torch.cat([torch.FloatTensor(s) for s in batch_next_state])

        expected_q_values = batch_reward
        return (batch_state, batch_age, batch_action, batch_next_state, batch_done, expected_q_values)


    def train(self, type):
        if len(self.memory) < self.BATCH_SIZE:
            return 1
        processes = []
        for _ in range(self.num_cpu):
            data = self.get_data()
            p = mp.Process(target=self.opt, args=(data, self.lock, self.queue, type))
            p.start()
            processes.append(p)
        for p in processes:
            loss = self.queue.get() # will block
            self.losses.append(loss)
        for p in processes:
            p.join()

        return 0

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    hidden = 16
    def __init__(self, num_features, num_actions):
        super(DQN, self).__init__()
        self.l1 = nn.Conv2d(1, self.hidden, 3) # 3
        self.l2 = nn.Conv2d(self.hidden, self.hidden, 3) # 5
        self.l3 = nn.Conv2d(self.hidden, self.hidden, 3) # 7
        self.l4 = nn.Conv2d(self.hidden, self.hidden, 3) # 9
        self.l5 = nn.Conv2d(self.hidden, self.hidden, 3) # 11
        self.out = nn.Linear(self.hidden + 1, num_actions)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x, age, relu=False):
        [N, a, b, c] = x.size()
        x = F.relu(self.l5(F.relu(self.l4(F.relu(self.l3(F.relu(self.l2(F.relu(self.l1(x))))))))))
        x = x.mean(-1).mean(-1)
        x = torch.cat([x, age], dim=1)
        out = self.out(x)
        return F.relu(out) if relu else out