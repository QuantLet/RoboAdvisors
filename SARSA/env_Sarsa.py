import numpy as np
import pandas as pd
import os

class UtilityEnv():
    """The environment for robo advisors to allocate portfolios according to investors' risk preference and market states
        without importance sampling, iterate data from month to month.
    """

    def __init__(self, df, month= 0):
        """Initialize attributes.
                    params:
                            df - the dataframe of states
                            count - the sampling number
        """
        self.df = df
        self.month = month
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

    def next_state(self, state):
        """select whether a disaster state happens according to disaster event probability p,
            then sample it
            hat_p: disaster event probabilites in different states
        """
        sample_state = np.random.choice(self.two_state_arr, 1, p=[0.9, 0.1])
        if sample_state[0] == 'disaster':
            sel_data_ = self.df[self.df['state'] == 'disaster'].sample(n=1)
        else:
            A = state == self.def_state("high")
            B = state == self.def_state("low")

            if A.all():  # s_t == high
                p = [0.7, 0.3]

            elif B.all():  # s_t == low
                p = [0.3, 0.7]

            else:  # s_t == disaster
                p = [0.5, 0.5]

            sample_state = np.random.choice(self.norm_state_arr, 1, p=p)  # etc. = array(['low'], dtype=object)
            sel_data_ = self.df[self.df['state'] == sample_state[0]].sample(n=1)

        data_ = sel_data_[["monthly_return", "standard_deviation", "rf"]]
        state_ = data_.to_numpy()
        state_ = state_[0]

        return state_

    def step(self, state, action, theta):
        """Sample next state state_, and observe reward - utility.
            params:
                state - current state
                action - the action that robo advisors take for the different types of investors
                theta - investors' risk preference
            output:
                state_ - next state
                reward - utility
                done - Ture if iterating to the last month
        """
        # reward = None


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

        terminal = self.month >= 74

        # Judge whether sample to the state
        if terminal:
            done = True
            state_ = "terminal"
            # reward = 0

        else:
            # load next state
            self.month += 1
            state_ = self.next_state(state)
            done = False

        return state_, reward, done


    def reset(self):
        """reset the initial state in each episode."""
        self.month = 0
        sel_data = self.df[self.df["month"] == self.month]
        data = sel_data[["monthly_return", "standard_deviation", "rf"]]
        state = data.to_numpy()
        state = state[0]
        return state


