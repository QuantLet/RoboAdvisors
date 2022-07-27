import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import time
import os


class UtilityDisasterEnv():
    """The environment with high, low, disaster state and the reward is to maximize the investors' utility."""

    def __init__(self, df, count = 0):
        """Initialize attributes.
            params:
                    df - the dataframe of states
                    count - the sampling number
        """
        self.df = df
        self.count = count
        self.two_state_arr = ['normal', 'disaster']
        self.state_arr = ['high', 'low', 'disaster']
        self.norm_state_arr = ['high', 'low']
        self.D_df = self.df[self.df["state"] == "disaster"]
        self.L_df = self.df[self.df["state"] == "low"]
        self.H_df = self.df[self.df["state"] == "high"]

    def def_state(self, state_str):
        df = self.df[self.df["state"] == state_str]
        data = df.sample(n=1)
        data = data[["monthly_return", "standard_deviation", "rf"]]
        state = data.to_numpy()
        state = state[0]

        return state

    def next_state(self, hat_p, state):
        """select whether a disaster state happens according to disaster event probability p,
            then sample it
            hat_p: a vector of disaster event probabilites in different states
        """
        sample_state = np.random.choice(self.two_state_arr, 1, p=[1-hat_p, hat_p])
        if sample_state[0] == 'disaster':
            sel_data_ = self.df[self.df['state'] == 'disaster'].sample(n=1)
        else:
            A = state == self.def_state("high")
            B = state == self.def_state("low")

            if A.all():  # s_t == high
                p = [0.7, 0.3]

            elif B.all():  # s_t == low
                p = [0.3, 0.7]

            else: #s_t == disaster
                p = [0.5, 0.5]

            sample_state = np.random.choice(self.norm_state_arr, 1, p=p)  # etc. = array(['low'], dtype=object)
            sel_data_ = self.df[self.df['state'] == sample_state[0]].sample(n=1)

        data_ = sel_data_[["monthly_return", "standard_deviation", "rf"]]
        state_ = data_.to_numpy()
        state_ = state_[0]

        return state_


    def step(self, state, action, theta, hat_p):
        """get next state, and reward.
            params:
                state - current state
                action - the action that robo advisors take for the different types of investors
                theta - investors' risk preference
                hat_p: disaster event probabilites in different states
            output:
                state_ - next state
                reward - utility
        """
        # the weight of risky asset in a portfolio
        weight = action

        # calculate portfolio return
        # = weight * risky return + (1-weight) * risk-free return
        risky_return = state[0]
        rf = state[2]
        portfolio_return = weight * risky_return + (1 - weight) * rf

        # calculate variance
        # = (weight * standard deviation) **2
        variance = (weight * state[1]) ** 2

        # utility model: mean-risk
        # utility = expectation - theta * variance
        utility = portfolio_return - theta * variance

        reward = utility

        # sampling 75 times in each episode
        terminal = self.count >= 74
        # print('count', self.count)
        
        # Judge whether sample to the terminal state
        if terminal:
            done = True
            state_ = "terminal"

        else:
            done = False
            self.count += 1
            # load next state
            state_ = self.next_state(hat_p, state)

        return state_, reward, done


    def reset(self):
        """reset the initial state in each episode."""
        self.count = 0
        sel_data = self.df[self.df["month"] == 0]
        data = sel_data[["monthly_return", "standard_deviation", "rf"]]
        state = data.to_numpy()
        state = state[0]

        return state



