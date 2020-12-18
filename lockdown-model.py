import pandas as pd
from itertools import combinations
from functools import reduce
import random
import numpy as np
import math
import matplotlib.pyplot as plt

NUM_AGENTS         = 20
MAX_TIMESTEPS      = 10
                   
MEAN_SOC           = 3.84/12
VAR_SOC            = 1.18/12
TIME               = 0
ROUNDS             = 16
RISK_FACTOR_DIST   = [0.01]*2 + [0.25]*8 + [0.5]*25 + [0.75]*35 + [1]*30
OCCUPATION_CLASSES = [(9/9)]*3 + [(6.5/9)]*1 + [(7.5/9)]*11 + [(2.1/9)]*4 + [(1.8/9)]*10 + [(0.8/9)]*3 + [(1.3/9)]*2 + [(1.1/9)]*3 + [(1/9)]*86
MIN_MEMBERS        = 1
MAX_MEMBERS        = 5
INFECTION_RATES    = [(0.9/1), (1.0/1), (1.5/1), (2.0/1), (4.0/1), (1.2/1), (1.1/1), (2.0/1)]*3
DECAY_STRETCH      = 15

def overall_exposure(p_values):
    p_values = list(map(float, p_values))
    if len(p_values) == 1:
           return p_values[0]
    overall_prob = sum(p_values)
    for i in range(1, len(p_values)):
        combos = list(combinations(p_values, i + 1))
        for comb in combos:
            overall_prob += pow(-1, i)*reduce((lambda x, y: x*y), comb)

    return overall_prob

class Household:
    social_eagerness = 0
    exposure_chance = 0
    risk_factor = 0
    n = 0
    
    def __init__(self, s, r, o):
        self.social_eagerness = s
        self.exposure_chance = sum(o)/len(o)
        self.risk_factor = r
        self.n = len(o)

    def value(self):
        #return self.risk_factor/(self.exposure_chance*self.social_eagerness)
        return 0

    def coalition_payoff(self, coalition):
        members = coalition.members
        if len(members) == 1 and members[0] == self:
            return self.value()
        else:
            exposures_list = [o.exposure_chance for o in list(set().union(members, [self]))]
            coalition_exposure = sum(exposures_list)
            #return pow(self.social_eagerness,
            #           self.risk_factor) / pow(coalition_exposure,
            #                                   1 - self.risk_factor)
            return self.social_eagerness - (coalition_exposure*self.risk_factor*decay()*infection())

    def __str__(self):
        return '<' + str(round(self.social_eagerness, 1)) + 'S, ' + str(round(self.risk_factor, 1)) + 'R, ' + str(round(self.exposure_chance, 1)) + 'E>'

class Coalition:
    members = []
    idnum = -1

    def __init__(self, m, i):
        self.members = m
        self.idnum = i

    def __str__(self):
        return str(list(map(str, self.members)))+'\n'
    
class World:
    agent_set = []
    coalition_set = []

    def __init__(self):
        self.agent_set = []
        for i in range(0, NUM_AGENTS):
            self.agent_set.append(Household(np.random.normal(loc=MEAN_SOC, scale=VAR_SOC),
                                            np.random.choice(RISK_FACTOR_DIST),
                                            np.random.choice(OCCUPATION_CLASSES, size = random.randint(MIN_MEMBERS,MAX_MEMBERS))))
        self.coalition_set = []
        for i in range(0, len(self.agent_set)):
            self.coalition_set.append(Coalition([self.agent_set[i]], i))

    def move_to(self, agent, coalition_id):
        for c in self.coalition_set:
            if c.idnum == coalition_id:
                if agent not in c.members:
                    c.members.append(agent)
            elif agent in c.members:
                c.members.remove(agent)
        return None

    def current_coalition(self, agent):
        for c in self.coalition_set:
            if agent in c.members:
                return c
        return None
    
    def best_coalition(self, agent):
        active_coalitions = []
        for c in self.coalition_set:
            if c.members:
                active_coalitions.append(c)
        #active_coalitions = self.coalition_set
        best_c = -1
        max_payoff = -1
        for c in active_coalitions:
            cur_payoff = agent.coalition_payoff(c)
            #print(cur_payoff)
            if cur_payoff > max_payoff:
                best_c = c
                max_payoff = cur_payoff
        #print('--------------------')
        if best_c == -1:
            return self.current_coalition(agent)
        else:
            if max_payoff < 0:
                for c in self.coalition_set:
                    if not c.members:
                        return c
            return best_c

    def __str__(self):
        result = 'Coalition sizes: ( '
        for c in self.coalition_set:
            if len(c.members):
                result += str(len(c.members)) + ' '
        result += ')\n'
        '''
        result += 'Coalition list:\n\n'
        for c in list(map(str, self.coalition_set)):
            if len(c) > 3:
                result += c

        result += str(infection()) + ' ' + str(decay()) + '\n'
        '''     
        return result
    
    def simulate(self):
        move_count = 0
        for i in range(0, MAX_TIMESTEPS):
            # simulation main loop
            random.shuffle(self.agent_set)
            for a in self.agent_set:
                best = self.best_coalition(a).idnum
                cur = self.current_coalition(a).idnum
                if best != cur:
                    self.move_to(a, best)
                    #print("agent moved")
                    move_count += 1
                # determine coalition with the highest value
                # compare highest value to self.
                # if self is best, leave coalition/stay alone
                # if a coalition is best, join it/stay if already there
            #print('-------------------------------')
        
def decay():
    return 1 - 1/math.sqrt(1+ 0.5*(math.exp(-16*(TIME/DECAY_STRETCH) + 12)))

def infection():
    return INFECTION_RATES[TIME]

# test cases for exposure function
assert overall_exposure([0.5, 0.2]) == 0.6
assert overall_exposure([0.5, 0.2, 0.3]) == 0.72

#forming coalitions in each round
world_1 = World()
coalition_avg_sizes = []
num_coalitions = []
for i in range(ROUNDS):
    TIME = i
    world_1.simulate()
    print(world_1)
    sum_sizes = 0
    coalition_count = 0
    for c in world_1.coalition_set:
        if c.members:
            coalition_count += 1
            sum_sizes += len(c.members)
    num_coalitions.append(coalition_count)
    coalition_avg_sizes.append(sum_sizes/coalition_count)
    print('-------------------------------')


plt.plot(coalition_avg_sizes, label="Average coalition size")
#plt.savefig("coalition_avg_sizes.png")
plt.plot(num_coalitions, label="Number of coalitions")
plt.plot(INFECTION_RATES[:len(num_coalitions)], label="Infection Risk Factor")
plt.legend()
plt.xlabel('Time (months)')
plt.savefig("num_coalitions.png")
