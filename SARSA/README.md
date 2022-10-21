[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **SARSA** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: 'SARSA'

Published in: 'RoboAdvisors'

Description: 'A conventional SARSA algorithm to optimize the policy in an epsilon-greedy way'

Keywords: 'SARSA, policy control, rewards'

Author: 'Jiawen Liang, Cathy Chen, Bowei Chen'

Submitted:  '27. 07. 2022'


```

### PYTHON Code
```python

from collections import defaultdict
import numpy as np
import pandas as pd


class Sarsa():
    """the tabular case TD(lambda) without importance sampling algorithms."""

    def __init__(self, action_space, gamma = 1, lam = 0.9):
        """Inherit class RL and its attributes.
                    params:
                            action_space - investors' actions
                            alpha - learning rate
                            gamma - discount rate
                            lambda - the rate between Sarsa and Monte Carlo
        """
        self.actions = action_space
        self.gamma = gamma
        self.lam = lam
        self.q_table = pd.DataFrame(columns=action_space, dtype=np.float64)
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

    def choose_action(self, state, epsilon):
        """With probability 1 − epsilon choose the greedy action
            With probability epsilon choose an action at random
        """
        self.check_state_exist(state)

        if np.random.rand() < 1 - epsilon:
            q_row = self.q_table.loc[state, :]
            optimal_actions = q_row[q_row == np.max(q_row)].index
            action = np.random.choice(optimal_actions)
        else:
            # choose action at random
            action = np.random.choice(self.actions)
        return action
    
    def learn(self, state, action, reward, state_, action_, alpha):
        """update the state value"""
        self.check_state_exist(state_)

        q_predict = self.q_table.loc[state,action]
        
        if state_ != "terminal":
            q_target = reward + self.gamma * self.q_table.loc[state_, action_]  # next state is not terminal
        else:
            q_target = reward
        error = q_target - q_predict
        
        self.eligibility_trace.loc[state, :] *= 0
        self.eligibility_trace.loc[state,action] = 1
        
        self.q_table += alpha * self.eligibility_trace * error

        # decay eligibility trace after update
        self.eligibility_trace *= self.gamma * self.lam

        q_value = self.q_table.loc[state, action]

        return q_value



```

automatically created on 2022-10-21