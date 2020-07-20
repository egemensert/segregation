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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from agent import Agent
from mind import Mind

class Environment:
    def __init__(self, size, p_hunter, p_prey,
            prey_reward = 1, stuck_penalty = 1,
            death_penalty = 1, p_resurrection = 0.2,
            agent_max_age = 10, agent_range = 2, num_actions = 5,
            same = True, lock = None, name=None,
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

        self.max_iteration = max_iteration
        self.lock = lock
        if same:
            weights = self.A_mind.network.state_dict()
            self.B_mind.network.load_state_dict(weights)


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
            os.mkdir(str(self.name)+'/episodes')
            self.map, self.agents, self.loc_to_agent, self.id_to_agent = self._generate_map()
            self._set_initial_states()
            self.mask = self._get_mask()
            self.crystal = np.zeros((max_iteration, H, W, 4)) # type, age, tr,  id
            self.iteration = 0
        else:
            assert False, "There exists an experiment with this name."

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

        with open("%s/episodes/a_age.csv" % self.name, "a") as f:
            f.write("%s, %s, %s\n" % (self.iteration, a_ages, a_ids))

        with open("%s/episodes/b_age.csv" % self.name, "a") as f:
            f.write("%s, %s, %s\n" % (self.iteration, b_ages, b_ids))

        if self.iteration == self.max_iteration - 1:
            A_losses = self.A_mind.get_losses()
            B_losses = self.B_mind.get_losses()
            np.save("%s/episodes/a_loss.npy" % self.name, np.array(A_losses))
            np.save("%s/episodes/b_loss.npy" % self.name, np.array(B_losses))

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
        f = gzip.GzipFile('%s/crystal.npy.gz' % self.name, "w")
        np.save(f, self.crystal)
        f.close()

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
