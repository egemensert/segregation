 #!/usr/bin/env python -W ignore::DeprecationWarning
import os
import gzip
import math
import copy
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp


from multiprocessing import Queue, Lock

import matplotlib as mpl
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#seed = 1
#[f(seed) for f in [random.seed, np.random.seed, torch.manual_seed]]


def kl_div(p, q):
    return np.sum(p * np.log(p / (q + 1e-6) + 1e-6))


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
        """
        x = F.relu(self.l1(x))
        r = self.l2(x)
        return F.relu(r) if relu else r
        """
        [N, a, b, c] = x.size()
        x = F.relu(self.l5(F.relu(self.l4(F.relu(self.l3(F.relu(self.l2(F.relu(self.l1(x))))))))))
        x = x.mean(-1).mean(-1)
        x = torch.cat([x, age], dim=1)
        out = self.out(x)
        return F.relu(out) if relu else out

class Mind:
    BATCH_SIZE = 256
    GAMMA = 0.98
    EPS_START = 0.9999
    EPS_END = 0
    EPS_DECAY = 100000
    #EPS_DECAY = 100
    TAU = 0.05

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

    def get_input_size(self):
        return self.input_size

    def get_output_size(self):
        return self.num_actions

    def get_losses(self):
        return self.losses

    def decide(self, state, age, type):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                state = torch.FloatTensor([[state]], device=device)
                age = torch.FloatTensor([[age]], device=device)
                q_values = self.network(type * state, age)
                return q_values.max(1)[1].view(1, 1).detach().item()
        else:
            rand = [[random.randrange(self.num_actions)]]
            return torch.tensor(rand, device=device, dtype=torch.long).detach().item()

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
        """
        lock.acquire()
        net, target_net, opt = self.copy()
        lock.release()
        """

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
        """
        lock.acquire()

        self.losses.append(loss.item())
        for target_param, param in zip(self.network.parameters(), net.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer.load_state_dict(opt.state_dict())
        """
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.TAU * param.data + target_param.data * (1.0 - self.TAU))

        # lock.release()

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
        for rank in range(self.num_cpu):
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


class Agent:
    def __init__(self, id, loc, type, mind, p, max_time):
        self.id = id
        self.alive = True
        self.loc = loc
        self.type = type
        self.current_state = None
        self.action = None
        self.next_state = None
        self.p_void = p
        self.mind = mind
        self.input_size = mind.get_input_size()
        self.output_size = mind.get_output_size()
        self.age = 0
        self.max_time = max_time
        self.time_remaining = max_time
        self.decision = None

    def update(self, reward, done):
        assert self.action != None, 'No Action'
        assert reward != None, 'No Reward'
        self.mind.remember([[[self.current_state]], [self.get_time_remaining() / self.max_time], [self.action], [[self.next_state]], [reward], [done]])

        #loss = self.mind.train()

        self.action = None
        if not done:
            self.current_state, self.next_state = self.next_state, None
        else:
            self.current_state, self.next_state = None, None

    def get_losses(self):
        return self.mind.get_losses()

    def decide(self, state):
        self.action = self.mind.decide(state, self.get_time_remaining() / self.max_time, self.get_type())
        self.age += 1
        self.time_remaining -= 1
        return self.action

    def get_state(self):
        return self.current_state

    def get_age(self):
        return self.age

    def get_time_remaining(self):
        return self.time_remaining

    def get_id(self):
        return self.id

    def get_loc(self):
        return self.loc

    def get_type(self):
        return self.type

    def get_decision(self):
        assert self.decision != None, "Decision is requested without setting."
        return self.decision

    def set_decision(self, decision):
        self.decision = decision

    def clear_decision(self):
        self.decision = None

    def set_loc(self, loc):
        self.loc = loc

    def set_current_state(self, state):
        self.current_state = state

    def set_next_state(self, state):
        self.next_state = state

    def eat(self, eating_benefit):
        self.time_remaining += 1 + eating_benefit

    def reset(self):
        self.time_remaining = self.max_time
        self.age = 0

    def die(self, state, rew, manual=False):
        self.alive = False
        if not self.action:
            self.action = np.random.choice(np.arange(self.mind.num_actions))
        if not manual:
            self.next_state = state
            self.update(rew, True)
        self.set_loc(None)
        self.reset()

    def is_alive(self):
        return self.alive

    def respawn(self, loc):
        self.alive = True
        self.clear_decision()
        self.set_loc(loc)
        #self.current_state = state

class Environment:
    def __init__(self, size, p_hunter, p_prey,
            prey_reward = 1, stuck_penalty = 1,
            death_penalty = 1, p_resurrection = 0.2,
            agent_max_age = 10, agent_range = 2, num_actions = 5,
            same = True, lock = None, name=None, details_csv = "details.csv",
            max_iteration = 5000, boundary=False):
        (H, W) = self.size = size
        input_size = (2*agent_range + 1) ** 2
        self.boundary_exists = boundary
        if lock:
            self.lock = lock
        else:
            self.lock = Lock()

        self.A_mind = Mind(input_size, num_actions, self.lock, Queue())
        self.B_mind = Mind(input_size, num_actions, self.lock, Queue())
        #self.P_mind = Mind(input_size, num_actions, self.lock, Queue())

        self.max_iteration = max_iteration
        self.lock = lock
        if same:
            weights = self.A_mind.network.state_dict()
            self.B_mind.network.load_state_dict(weights)
            #self.P_mind.network.load_state_dict(weights)


        self.p_prey = p_prey
        self.p_resurrection = p_resurrection
        assert 1 - 2*p_hunter - p_prey > 0, 'Free space probability is less than one'
        self.probs = np.array([p_hunter, 1 - 2*p_hunter - p_prey, p_hunter, p_prey])

        self.prey_reward = prey_reward
        self.stuck_penalty = stuck_penalty
        self.death_penalty = death_penalty
        self.agent_max_age = agent_max_age
        self.hzn = agent_range
        self.vals = [-1, 0, 1, 2]
        self.names_to_vals = {"void": -2, "A": -1, "free": 0, "B": 1, "prey": 2}
        self.vals_to_names = {v: k for k, v in self.names_to_vals.items()}
        self.vals_to_index = {-1: 0, 0: 1, 1: 2, 2: 3}

        self.num_grids = size[0] * size[1]

        self.id_to_type = {}
        self.id_to_lives = {}

        self.crystal = np.zeros((max_iteration, H, W, 3)) # type, age, id
        self.history = []
        self.id_track = []
        self.records = []

        self.args = [self.prey_reward,
                self.stuck_penalty,
                self.death_penalty,
                self.agent_max_age,
                self.hzn]
        if name:
            self.name = name
        else:
            self.name = abs(hash(tuple([self] + self.args)))

        if not os.path.isdir(str(self.name)):
            os.mkdir(str(self.name))
            self.map, self.agents, self.loc_to_agent, self.id_to_agent = self._generate_map()
            self._set_initial_states()
            self.mask = self._get_mask()
            self.crystal = np.zeros((max_iteration, H, W, 4)) # type, age, tr,  id
            self.iteration = 0
        else:

            deads = np.load("%s/deads.npy" % self.name)
            prev_crystal = np.load("%s/crystal.npy" % self.name)
            self.iteration = len(prev_crystal)
            self.A_mind.load(self.name, "A", self.iteration)
            self.B_mind.load(self.name, "B", self.iteration)
            self.crystal = np.zeros((len(prev_crystal) + max_iteration, H, W, 4))
            self.crystal[:len(prev_crystal)] = prev_crystal
            self.map = prev_crystal[-1, :, :, 0]
            self.agents = []
            self.loc_to_agent = {}
            self.id_to_agent = {}
            for i, row in enumerate(self.map):
                for j, val in enumerate(row):
                    if not val == self.names_to_vals["free"]:
                        if val == self.names_to_vals["A"]:
                            mind = self.A_mind
                        elif val == self.names_to_vals["B"]:
                            mind = self.B_mind
                        elif val == self.names_to_vals["prey"]:
                            mind = self.P_mind
                        idx = prev_crystal[-1, i, j, 3]
                        agent = Agent(idx, (i, j), val, mind,  self.probs[1], self.agent_max_age)
                        agent.age = prev_crystal[-1, i, j, 1]
                        agent.time_remaining = prev_crystal[-1, i, j, 2]
                        self.loc_to_agent[(i, j)] = agent
                        self.id_to_agent[idx] = agent
                        self.agents.append(agent)

            self._set_initial_states()
            for dead in deads:
                if dead[0] == self.names_to_vals["A"]:
                    mind = self.A_mind
                elif dead[0] == self.names_to_vals["B"]:
                    mind = self.B_mind
                agent = Agent(dead[1], (-1, -1), dead[0], mind, self.probs[1], self.agent_max_age)
                agent.die([], 0, manual=True)
                self.agents.append(agent)

    def configure(self, prey, penalty, age):
        self.prey_reward = prey
        self.stuck_penalty = penalty
        self.agent_max_age = age

    def get_agents(self):
        return self.agents

    def get_map(self):
        return self.map.copy()

    def move(self, agent):
        (i, j) = loc = agent.get_loc()
        (i_n, j_n) = to = agent.get_decision()
        self.map[i, j] = 0
        self.map[i_n, j_n] = agent.get_type()
        agent.set_loc(to)
        self.loc_to_agent[to] = agent
        del self.loc_to_agent[loc]

    def step(self, agent, by):
        if agent.is_alive() and agent.get_time_remaining() == 0:
            rew = self.kill(agent)
        elif agent.is_alive():
            (i, j) = agent.get_loc() # current location
            assert self.loc_to_agent[(i, j)]
            (di, dj) = by
            (i_n, j_n) = self._add((i, j), (di, dj)) #new location
            agent.set_decision((i_n, j_n))
            if self.map[i_n, j_n] == self.names_to_vals["free"]:
                rew = self.on_free(agent)
                assert rew != None
            elif self.map[i_n, j_n] == self.names_to_vals["prey"]:
                prey = self.loc_to_agent[(i_n, j_n)]
                if agent.get_type() in [-1, 1]:
                    rew = self.on_prey(agent, prey)
                else:
                    rew = self.on_same(agent, prey)
                assert rew != None
            elif self.map[i_n, j_n] * agent.get_type() == -1:
                opponent = self.loc_to_agent[(i_n, j_n)]
                rew = self.on_opponent(agent, opponent)
                assert rew != None
            elif di == 0 and dj == 0:
                rew = self.on_still(agent)
            else:
                other_agent = self.loc_to_agent[(i_n, j_n)]
                rew = self.on_same(agent, other_agent)
                assert rew != None
            done = False
            self.update_agent(agent, rew, done)
            agent.clear_decision()
        else:
            resurrected = self.resurrect(agent)
            rew = 0
            if resurrected and agent.get_type() in [-1, 1]:
                assert agent.get_time_remaining() == self.agent_max_age, agent.get_time_remaining()
                assert agent.get_loc() != None, "wtf?"
        return rew

    def on_free(self, agent):
        raise NotImplementedError('Please implement what to'\
         'do when agent steps on a free grid.')

    def on_still(self, agent):
        raise NotImplementedError('Please implement what to'\
         'do when agent stands still.')

    def on_prey(self, agent, prey):
        raise NotImplementedError('Please implement what to'\
         'do when agent encounters a prey.')

    def on_obstacle(self, agent):
        raise NotImplementedError('Please implement what to'\
         'do when agent encounters an obstacle.')

    def on_same(self, agent, other):
        raise NotImplementedError('Please implement what to'\
         'do when agent encounters its same kind.')

    def on_opponent(self, agent, opponent):
        raise NotImplementedError('Please implement what to'\
         'do when agent encounters a different kind.')

    def kill(self, victim, killer=None):
        raise NotImplementedError('Please implement what to'\
         'do when victim dies (with or wo killer).')


    def resurrect(self, agent):
        resurrected = False
        if np.random.random() < self.p_resurrection:
            locs = [(a, b)  for a in range(len(self.map))
                for b in range(len(self.map[0]))
                if self.map[a,b] == 0]
            idx = np.random.choice(range(len(locs)))
            (i, j) = loc = locs[idx]
            self.map[i, j] = agent.get_type()
            self.loc_to_agent[loc] = agent
            agent.respawn(loc)
            agent.set_current_state(self.get_agent_state(agent))
            resurrected = True
        return resurrected

    def update_agent(self, agent, rew, done):
        state = self.get_agent_state(agent)
        agent.set_next_state(state)
        name = self.vals_to_names[agent.get_type()]
        agent.update(rew, done)
        return rew

    def dump(self):
        with open(self.details_csv+"_agents", "w") as f:
            f.write("id, type, vals\n")
            for k, vs in self.id_to_lives.items():
                typ = self.id_to_type[k]
                f.write("%s, %s, %s\n" % (k, typ, " ".join(map(str, vs))))

    def update(self):
        self.iteration += 1
        self.history.append(self.map.copy())

        self.A_mind.train(self.names_to_vals["A"])
        self.B_mind.train(self.names_to_vals["B"])
        a_ages = []
        a_ids = []
        b_ages = []
        b_ids = []
        id_track = np.zeros(self.map.shape)
        self.deads = []
        for agent in self.agents:
            typ = agent.get_type()
            age = agent.get_age()
            idx = agent.get_id()

            if agent.is_alive():
                i, j = agent.get_loc()
                tr = agent.get_time_remaining()
                id_track[i, j] = idx
                self.crystal[self.iteration - 1, i, j] = [typ, age, tr, idx]
            else:
                self.deads.append([agent.get_type(), agent.get_id()])

            type = agent.get_type()
            if type == self.names_to_vals["A"]:
                a_ages.append(str(age))
                a_ids.append(str(idx))
            else:
                b_ages.append(str(age))
                b_ids.append(str(idx))

        self.id_track.append(id_track)
        a_ages = " ".join(a_ages)
        b_ages = " ".join(b_ages)

        a_ids = " ".join(a_ids)
        b_ids = " ".join(b_ids)

        with open("episodes/%s_a_age.csv" % self.name, "a") as f:
            f.write("%s, %s, %s\n" % (self.iteration, a_ages, a_ids))

        with open("episodes/%s_b_age.csv" % self.name, "a") as f:
            f.write("%s, %s, %s\n" % (self.iteration, b_ages, b_ids))

        if self.iteration == self.max_iteration - 1:
            A_losses = self.A_mind.get_losses()
            B_losses = self.B_mind.get_losses()
            np.save("episodes/%s_a_loss.npy" % self.name, np.array(A_losses))
            np.save("episodes/%s_b_loss.npy" % self.name, np.array(B_losses))

    def shuffle(self):
        map = np.zeros(self.size)
        loc_to_agent = {}

        locs = [(i, j) for i in range(self.map.shape[0]) for j in range(self.map.shape[1]) if self.map[i, j] == 0]
        random.shuffle(locs)
        id_track = np.zeros(self.map.shape)
        for i, agent in enumerate(self.agents):
            loc = locs[i]
            agent.respawn(loc)
            loc_to_agent[loc] = agent
            map[loc] = agent.get_type()
            id_track[loc] = agent.get_id()


        self.map, self.loc_to_agent = map, loc_to_agent
        self._set_initial_states()
        self.history = [map.copy()]
        self.id_track = [id_track]
        self.records = []
        self.iteration = 0

    def record(self, rews):
        self.records.append(rews)

    def save(self, episode):
        history = np.array(self.history)
        id_track = np.array(self.id_track)
        #np.save("episodes/%s_%s.npy" % (episode, self.name) , history)
        #np.save("episodes/%s_%s_ids.npy" % (episode, self.name) , id_track)

        #self._to_csv(episode)
        #self.A_mind.save(self.name, "A")
        #self.B_mind.save(self.name, "B")
        f = gzip.GzipFile('%s/crystal.npy.gz' % self.name, "w")
        np.save(f, self.crystal)
        f.close()
        #np.save("%s/deads.npy" % self.name, np.array(self.deads))

    def save_agents(self):
        self.lock.acquire()
        pickle.dump(self.agents, open("agents/agent_%s.p" % (self.name), "wb" ))
        self.lock.release()

    def get_agent_state(self, agent):
        hzn = self.hzn
        i, j = agent.get_loc()
        fov = np.zeros((2 * hzn + 1, 2 *  hzn + 1)) - 2
        if self.boundary_exists:
            start_i, end_i, start_j, end_j = 0, 2 * hzn + 1, 0, 2 * hzn + 1
            if i < hzn:
                start_i = hzn - i
            elif i + hzn - self.size[0] + 1 > 0:
                end_i = (2 * hzn + 1) - (i + hzn - self.size[0] + 1)
            if j < hzn:
                start_j = hzn - j
            elif j + hzn - self.size[1] + 1 > 0:
                end_j = (2 * hzn + 1) - (j + hzn - self.size[1] + 1)
            i_upper = min(i + hzn + 1, self.size[0])
            i_lower = max(i - hzn, 0)

            j_upper = min(j + hzn + 1, self.size[1])
            j_lower = max(j - hzn, 0)

            fov[start_i: end_i, start_j: end_j] = self.map[i_lower: i_upper, j_lower: j_upper].copy()
        else:
            for di in range(-hzn, hzn+1):
                for dj in range(-hzn, hzn+1):
                    fov[hzn + di, hzn + dj] = self.map[(i+di) % self.size[0], (j+dj) % self.size[1]]

        fov[hzn, hzn] = agent.get_type()
        return fov

    def _to_csv(self, episode):
        with open("episodes/%s_%s.csv" % (episode, self.name), 'w') as f:
            f.write(', '.join(self.records[0].keys()) + '\n')
            proto = ", ".join(['%.3f' for _ in range(len(self.records[0]))]) + '\n'
            for rec in self.records:
                f.write(proto % tuple(rec.values()))

    def _add(self, fr, by):
        i, j = fr
        di, dj = by
        if self.boundary_exists:
            i_n = min(max(i + di, 0), self.size[0] - 1)
            j_n = min(max(j + dj, 0), self.size[1] - 1)
        else:
            i_n = (i + di) % self.size[0]
            j_n = (j + dj) % self.size[1]
        return (i_n, j_n)

    def _get_mask(self):
        mask = []
        for i, row in enumerate(self.map):
            foo = []
            for j, col in enumerate(row):
                foo.append((-1) ** (i + j))
            mask.append(foo)
        return np.array(mask)

    def _generate_map(self):
        map = np.zeros(self.size)
        loc_to_agent = {}
        id_to_agent = {}
        agents = []
        idx = 0
        for i, row in enumerate(map):
            for j, col in enumerate(row):
                val = np.random.choice(self.vals, p=self.probs)
                if not val == self.names_to_vals["free"]:
                    if val == self.names_to_vals["A"]:
                        mind = self.A_mind
                    elif val == self.names_to_vals["B"]:
                        mind = self.B_mind
                    else:
                        assert False, 'Error'
                    agent = Agent(idx, (i, j), val, mind,  self.probs[1], self.agent_max_age)
                    loc_to_agent[(i, j)] = agent
                    id_to_agent[idx] = agent
                    agents.append(agent)
                    idx += 1
                map[i, j] = val
        return map, agents, loc_to_agent, id_to_agent

    def predefined_initialization(self, file):
        with open(file) as f:
            for i, line in enumerate(f):
                if not i:
                    keys = [key.strip() for key in line.rstrip().split(',')]
                line.rstrip().split(',')

    def _set_initial_states(self):
        for agent in self.agents:
            state = self.get_agent_state(agent)
            agent.set_current_state(state)

    def _count(self, arr):
        cnt = np.zeros(len(self.vals))
        arr = arr.reshape(-1)
        for elem in arr:
            if elem in self.vals_to_index:
                cnt[self.vals_to_index[elem]] += 1
        return cnt
