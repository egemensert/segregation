import numpy as np

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