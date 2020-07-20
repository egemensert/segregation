 #!/usr/bin/env python -W ignore::DeprecationWarning

from environment import Environment

from itertools import count
from multiprocessing import Process, Lock

import time
import random
import os, sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import numpy as np

class Schelling(Environment):
    def __init__(self, size, p_hunter = 0.05, p_prey = 0,
            prey_reward = 1, stuck_penalty = 1,
            death_penalty = 1, p_resurrection = 0.2,
            agent_max_age = 100, agent_range = 2, num_actions = 5,
            same = True, lock = None,
            max_iteration = 5000, name = None, eating_bonus = 1,
            alpha=1., beta=1., gamma = 1.):

        super(Schelling, self).__init__(size, p_hunter, p_prey,
                prey_reward = prey_reward, stuck_penalty = stuck_penalty,
                death_penalty = death_penalty, p_resurrection = p_resurrection,
                agent_max_age = agent_max_age, agent_range = agent_range, num_actions = num_actions,
                same = same, lock = lock, name = name, max_iteration = max_iteration)

        self.eating_bonus = prey_reward

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.alive_reward = 0.1

    def default(self, agent):
        curr = self.get_agent_state(agent)
        prev = agent.get_state()
        default = (agent.get_type() * (curr - prev))
        sames = default[default > 0.1].sum()
        diffs = self.alpha * default[default < -0.1].sum()
        return sames + diffs

    def on_free(self, agent):
        self.move(agent)
        return self.beta * self.alive_reward + self.default(agent)

    def on_opponent(self, agent, opponent):
        _ = self.kill(opponent, killer=agent)
        return self.beta * self.alive_reward + self.default(agent) + self.prey_reward * self.gamma

    def on_still(self, agent):
        return -10*self.alive_reward

    def on_obstacle(self, agent):
        return -10*self.alive_reward
    def on_same(self, agent, other):
        return -10*self.alive_reward

    def kill(self, victim, killer=False):
        if victim.get_type() in [-1, 1]:
            id = victim.get_id()
            if id in self.id_to_type:
                self.id_to_lives[id].append(victim.get_age())
            else:
                self.id_to_type[id] = victim.get_type()
                self.id_to_lives[id] = [victim.get_age()]
        i, j = victim.get_loc()

        self.map[i, j] = 0
        state = self.get_agent_state(victim)
        del self.loc_to_agent[(i, j)]

        victim.die(state, -self.death_penalty)
        if killer:
            killer.eat(self.gamma * 1)
            self.move(killer)
        return -self.death_penalty

def play(map, episodes, iterations, eps=1e-6):
    # map.configure(prey_reward, stuck_penalty, agent_max_age)
    agents = map.get_agents()
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1), (0, 0)]
    times = 0
    for episode in range(episodes):
        c = 0
        for t in count():
            t_start = time.time()
            state = map.get_map()
            random.shuffle(agents)

            keys = ["A", "B", "prey"]
            rews = {key: 0 for key in keys}
            counts = {key: 0 for key in keys}
            for agent in agents:
                towards = None
                name = map.vals_to_names[agent.get_type()]
                if agent.is_alive():
                    agent_state = agent.get_state()
                    action = agent.decide(agent_state)
                    towards = directions[action]
                rew = map.step(agent, towards)
                rews[name] += rew
                counts[name] += 1

            map.update()

            map.record(rews)

            next_state = map.get_map()

            time_elapsed = time.time() - t_start
            times += time_elapsed
            avg_time = times / (t + 1)
            print("I: %d\tTime Elapsed: %.2f" % (t+1, avg_time), end='\r')
            if abs(next_state - state).sum() < eps:
                c += 1

            if t == (iterations - 1) or c == 20:
                break

            state = next_state
        map.save(episode)
    print("SIMULATION IS FINISHED.")

if __name__ == '__main__':
    [_, name, iterations, agent_range, prey_reward, max_age, alpha, beta, gamma] = sys.argv

    # alpha is for schelling reward
    # beta is for vigilance reward
    # gamma is for interdependence reward

    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)

    episodes = 1
    iterations = int(iterations)
    l = Lock()

    args = ["Name",
            "Prey Reward",
            "Stuck Penalty",
            "Death Penalty",
            "Agent Max Age",
            "Agent Field of View"]

    society = Schelling

    play(society((50, 50), agent_range = int(agent_range),
        prey_reward = int(prey_reward), name=name,
        agent_max_age = int(max_age), max_iteration = int(iterations),
        lock=l, alpha=float(alpha), beta=float(beta), gamma=float(gamma)), 1, iterations)
