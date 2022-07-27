from collections import defaultdict
import numpy as np
import pandas as pd
import os
import ast
import math

class Sarsa_IS():
    """the tabular case Sarsa(lambda) with importance sampling algorithms."""

    def __init__(self, df, action_space, p, delta=0.01, gamma = 1, lam = 0.9):
        """inherit class RL and its attributes.
                    params:
                            df - DataFrame used in this learning
                            action_space - investors' actions
                            p - true rare-event probabilities
                            delta - number to restrict the boundary of estimated rare-event probabilities
                            gamma - discount rate
                            lambda - the rate between Sarsa and Monte Carlo
                """
        self.df = df
        self.actions = action_space
        self.p = p
        self.delta = delta
        self.gamma = gamma
        self.lam = lam

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.d_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.n_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state
                    )
            self.q_table = self.q_table.append(to_be_append)

            # also update eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)
            
    def check_disaster_state_exist(self, state):
        if state not in self.d_table.index:
            self.d_table = self.d_table.append(pd.Series(
                                                [1] * len(self.actions),
                                                index=self.d_table.columns,
                                                name=state
                                                ))

    def check_normal_state_exist(self, state):
        if state not in self.n_table.index:
            self.n_table = self.n_table.append(pd.Series(
                [0] * len(self.actions),
                index=self.n_table.columns,
                name=state
            ))

    def choose_action(self, state, epsilon):
        self.check_state_exist(state)

        if np.random.rand() < 1 - epsilon:
            q_row = self.q_table.loc[state, :]
            optimal_actions = q_row[q_row == np.max(q_row)].index
            action = np.random.choice(optimal_actions)
            # print("action", action)
        else:
            action = np.random.choice(self.actions)
            
        return action

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
        # print('hat_p', hat_p)
        return w


    def learn(self, state, action, reward, state_, action_, w, alpha):
        """update the state action value"""
        self.check_state_exist(state_)
        self.check_disaster_state_exist(state)
        self.check_normal_state_exist(state)
        self.check_disaster_state_exist(state_)
        self.check_normal_state_exist(state_)

        q_predict = self.q_table.loc[state,action]
        # print("q_predict",q_predict)
        # print("state", state, "previous value", q_predict)
        
        if state_ != "terminal":
            q_target = w * (reward + self.gamma * self.q_table.loc[state_, action_])  # next state is not terminal
        else:
            q_target = reward
        error = q_target - q_predict

        self.eligibility_trace.loc[state, :] *= 0
        self.eligibility_trace.loc[state,action] = 1

        if self.q_table.isnull().values.any():
            print('Oops! There is Nan in q_table before updating')
            print('Nan amount', self.q_table.isnull().sum().sum())
            print('q value', self.q_table.loc[state, action])

        self.q_table += alpha * self.eligibility_trace * error

        self.eligibility_trace *= w * self.gamma * self.lam
        q_value = self.q_table.loc[state, action]
                
        return q_value


    def update_RE_prob(self, state, action, reward, state_, action_, w, alpha):
        """update proposed disaster event probability in each step"""
        if state_ == self.disaster_set():
            d_predict = self.d_table.loc[state, action]
            if state_ != "terminal":
                d_target = w * (reward + self.gamma * self.d_table.loc[state_, action_])  # next state is not terminal
            else:
                d_target = reward
            error = d_target - d_predict
            self.d_table += alpha * self.eligibility_trace * error
        else:
            n_predict = self.n_table.loc[state, action]
            if state_ != "terminal":
                n_target = w * (reward + self.gamma * self.n_table.loc[state_, action_])  # next state is not terminal
            else:
                n_target = reward
            error = n_target - n_predict
            self.n_table += alpha * self.eligibility_trace * error


        fraction = abs(self.d_table.loc[state, action]) / (abs(self.d_table.loc[state, action]) + abs(self.n_table.loc[state, action]))
        hat_p = min(max(self.delta, fraction), (1 - self.delta))

        return hat_p



