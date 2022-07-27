from collections import defaultdict
import numpy as np
import pandas as pd
import os
import ast
import math

class DEIS():
    """the tabular case Sarsa(lambda) with importance sampling algorithms."""

    def __init__(self, df, policy, p, delta=0.01, gamma = 1, lam = 0.9):
        """inherit class RL and its attributes.
                    params:
                            df - DataFrame used in this learning
                            policy - the prob of investors' action in a state
                            p - true rare-event probabilities
                            delta - number to restrict the boundary of estimated rare-event probabilities
                            gamma - discount rate
                            lambda - the rate between Sarsa and Monte Carlo
                """
        self.df = df
        self.policy = policy
        self.p = p
        self.delta = delta
        self.gamma = gamma
        self.lam = lam

        self.V = dict()
        self.V_d = dict()
        self.V_dc = dict()
        self.eligibility_trace = dict()

    def check_state_exist(self, state):
        if state not in self.V.keys():
            self.V[state] = 0
            self.eligibility_trace[state] = 0
            
    def check_disaster_state_exist(self, state):
        if state not in self.V_d.keys():
            self.V_d[state] = 0

    def check_normal_state_exist(self, state):
        if state not in self.V_dc.keys():
            self.V_dc[state] = 0

    def choose_action(self, state):
        self.check_state_exist(str(state))

        return self.policy[state]

    def disaster_set(self):
        """define the disaster event set"""
        D_df = self.df[self.df["state"] == "disaster"]
        data = D_df.sample(n=1)
        data = data[["monthly_return", "standard_deviation", "rf"]]
        state = data.to_numpy()
        state = state[0]
        return str(state)

    def IS_weight(self, state_, hat_p):
        """compute the importance sampling weight"""
        if state_ == self.disaster_set():
            w = self.p / hat_p
        else:
            w = (1 - self.p) / (1 - hat_p)
        return w


    def learn(self, state, reward, state_, w, alpha):
        """update the state action value"""
        self.check_state_exist(state_)
        self.check_disaster_state_exist(state)
        self.check_normal_state_exist(state)
        self.check_disaster_state_exist(state_)
        self.check_normal_state_exist(state_)

        v_predict = self.V[state]
        
        if state_ != "terminal":
            v_target = w * (reward + self.gamma * self.V[state_] )  # next state is not terminal
        else:
            v_target = reward
        error = v_target - v_predict
        # print('error',error)

        self.eligibility_trace[state] = 1

        for s in self.V.keys():
            self.V[s] += alpha * self.eligibility_trace[s] * error

            self.eligibility_trace[s] *= self.gamma * self.lam

        state_value = self.V[state]

        return state_value


    def update_RE_prob(self, state, reward, state_, w, alpha):
        """update proposed disaster event probability in each step"""
        if state_ == self.disaster_set():
            self.V_d[state] = (1 - alpha) * self.V_d[state] + alpha * self.p * (
                    reward + self.gamma * self.V[state])

        else:
            self.V_dc[state] = (1 - alpha) * self.V_dc[state] + alpha * (1 - self.p) * (
                                                          reward + self.gamma * self.V[state])

        fraction = abs(self.V_d[state]) / (abs(self.V_d[state]) + abs(self.V_dc[state]))
        hat_p = min(max(self.delta, fraction), (1 - self.delta))

        return hat_p


