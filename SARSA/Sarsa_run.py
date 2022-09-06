import numpy as np
import pandas as pd
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
# print(matplotlib.get_backend())
# import time

from env_Sarsa import UtilityEnv
from Sarsa import Sarsa

def train_Sarsa(max_episodes):
    """Run episodes and train the agent for investors.
        params:
                max_episodes - the maximum number of episodes

        return:
                policy_df - actions
                reward_df - the rewards
                q_df - q values
    """
    policy_df = pd.DataFrame()
    reward_df = pd.DataFrame()
    q_df = pd.DataFrame()

    for episode in range(max_episodes):
        print("Sarsa episode", episode)

        # select the initial state
        state = environment.reset()

        # decreasing epsilon over time
        epsilon = 1 / (episode + 10)

        # learning rate is approaching to 0 when episodes run
        alpha = 1 / (episode + 10)

        # RL choose action based on observation
        action = model.choose_action(str(state), epsilon)

        # initial all zero eligibility trace
        model.eligibility_trace *= 0

        step_policy_df = pd.DataFrame()
        step_q_df = pd.DataFrame()
        step_reward_df = pd.DataFrame()

        while True:
            # RL take action and get next observation and reward
            state_, reward, done= environment.step(state, action, theta)

            # RL choose action based on next observation
            action_ = model.choose_action(str(state_), epsilon)

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            q_value = model.learn(str(state), action, reward, str(state_), action_, alpha)

            # memory the optimal policy to step_policy_df
            step_policy = pd.DataFrame({"episode": episode, "state": str(state), "action": action}, index = [0])
            step_policy_df = step_policy_df.append(step_policy, ignore_index= True)

            # memory the step rewards to step_reward_df
            step_reward = pd.DataFrame({"episode": episode, "state": str(state), "rewards": reward}, index=[0])
            step_reward_df = step_reward_df.append(step_reward, ignore_index=True)

            # memory the q_value
            step_q = pd.DataFrame({"episode": episode, "state": str(state), "q_value": q_value}, index=[0])
            step_q_df = step_q_df.append(step_q, ignore_index=True)

            # let next state be current state in next step
            state = state_
            action = action_

            # To the last step
            if done:
                break

        # memory policy episode by episode
        policy_df = pd.concat([policy_df, step_policy_df])

        # memory rewards episode by episode
        reward_df = pd.concat([reward_df, step_reward_df])

        # memory q episode by episode
        q_df = pd.concat([q_df, step_q_df])

    return policy_df, reward_df, q_df


if __name__ == '__main__':
    path = "covid_data.csv"
    # path = "financial_crisis_data.csv"

    df = pd.read_csv(path)

    # investor type: general
    action_space = np.arange(1, 101) / 100

    # risk preference, take the mean {2.2, 2.3, ... , 8.4}
    theta = np.mean(np.arange(22, 84) / 10)

    environment = UtilityEnv(df)

    model = Sarsa(action_space)
    #
    policy_df, reward_df, q_df = train_Sarsa(500)
    # #
    # policy_df.to_pickle('Sarsa_policy_crisis.pkl')
    # reward_df.to_pickle('Sarsa_rewards_crisis.pkl')
    # q_df.to_pickle('Sarsa_q_crisis.pkl')

    policy_df.to_pickle('Sarsa_policy_covid.pkl')
    reward_df.to_pickle('Sarsa_rewards_covid.pkl')
    q_df.to_pickle('Sarsa_q_covid.pkl')
    # #
    # q_df = pd.read_pickle('Sarsa_q_covid.pkl')
    # reward_df = pd.read_pickle('Sarsa_rewards_covid.pkl')
    # policy_df = pd.read_pickle(
    #     'Sarsa_policy_covid.pkl')

  
