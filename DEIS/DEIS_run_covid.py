import numpy as np
import pandas as pd
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import pickle
from matplotlib.pyplot import MultipleLocator
# print(matplotlib.get_backend())
# import time

from env_DEIS import UtilityDisasterEnv
from DEIS import DEIS


"""
methods: 
    DEIS - REASA model to evaluate the policy

objectives:
    to see which policy is better
    to see if importance sampling works
"""


def train_DEIS(max_episodes, model):
    """Run episodes and train the agent (robo advisors) for investors.
        params:
                max_episodes - the maximum number of episodes
                modeo - DEIS

        return:
                v_df - state values
    """
    # start_time = time.time()
    hat_p_list = []
    v_df = pd.DataFrame()

    # Initialized proposed disaster event probability
    # hat_p = dict({s_high: 0.9, s_low: 0.9, s_disaster: 0.9})
    hat_p = 0.8

    for episode in range(max_episodes):
        print("REASA episode", episode)

        # select the initial state
        state = environment.reset()

        # learning rate is approaching to 0 when episodes run longer
        alpha = 1 / (episode + 1)

        # initial eligibility trace e(s0)=1 and e(s)=0
        model.eligibility_trace = {k: 0 for k in model.eligibility_trace.keys()}
        model.eligibility_trace[str(state)] = 1

        step_hat_p_list = []
        step_policy_df = pd.DataFrame()
        step_reward_df = pd.DataFrame()
        step_v_df = pd.DataFrame()

        while True:
            action = model.choose_action(str(state))
            # RL take action and get next observation and reward
            state_, reward, done = environment.step(state, action, theta, hat_p)
            # print("state_", state_)

            # compute the imporance sampling weight
            w = model.IS_weight(str(state_), hat_p)

            # TD update
            v_value = model.learn(str(state), reward, str(state_), w, alpha)

            # update proposed disaster event probability.
            hat_p = model.update_RE_prob(str(state), reward, str(state_), w, alpha)
            step_hat_p_list.append(hat_p)
            # print('hat_p', hat_p[str(state)])

            # memory the optimal policy to step_policy_df
            step_policy = pd.DataFrame({"episode": episode, "state": str(state), "action": action}, index=[0])
            step_policy_df = step_policy_df.append(step_policy, ignore_index=True)

            # memory the step rewards to step_reward_df
            reward_adjust = reward
            step_reward = pd.DataFrame({"episode": episode, "state": str(state), "rewards": reward_adjust}, index=[0])
            step_reward_df = step_reward_df.append(step_reward, ignore_index=True)

            # memory the q_value
            step_v = pd.DataFrame({"episode": episode, "state": str(state), "state_value": v_value}, index=[0])
            # print(step_q)
            step_v_df = step_v_df.append(step_v, ignore_index=True)
            # print(step_q_df)

            # let next state be current state in next step
            state = state_

            # To the last step
            if done:
                # np.save("results/growth_REASA_policy",policy_df)
                hat_p_mean = sum(step_hat_p_list) / len(step_hat_p_list)
                hat_p_list.append(hat_p_mean)
                break

        # memory q episode by episode
        v_df = pd.concat([v_df, step_v_df])

    return v_df


if __name__ == '__main__':
    path = "/Users/liangjiawen/PycharmProjects/TDlearning/preprocessor/covid_data_new.csv"
    df = pd.read_csv(path)

    # investor type: general
    action_space = np.arange(1, 101) / 100
    # risk preference, take the mean {2.2, 2.3, ... , 8.4}
    theta = np.mean(np.arange(22, 84) / 10)

    # True rare-event probabilites
    s_high = str(np.array([0.0125, 0.05,  0.002]))
    s_low = str(np.array([0.005, 0.03, 0.002]))
    s_disaster = str(np.array([-0.0275, 0.1352, 0.002]))
    # s_disaster = str(np.array([-0.03, 0.06, 0.002]))

    # # covid policy
    policy_IS = dict({s_high: 0.24, s_low: 0.59, s_disaster: 0.01})
    policy_Sarsa = dict({s_high: 0.3, s_low: 0.69, s_disaster: 0.14})

    p = 0.1

    environment = UtilityDisasterEnv(df)

    model_IS = DEIS(df, policy_IS, p)
    model_Sarsa = DEIS(df, policy_Sarsa, p)

    # # train REASA model using Sarsa optimal policy
    v_IS = train_DEIS(2000, model_IS)

    v_IS.to_pickle('covid_DEIS_IS_v.pkl')

    v_S = train_DEIS(2000, model_Sarsa)
    v_S.to_pickle('covid_DEIS_S_v.pkl')


    #
    # v_IS = pd.read_pickle("crisis_DEIS_IS_v.pkl")
    # v_S = pd.read_pickle("crisis_DEIS_S_v.pkl")
    #
    #
    # def get_value(df):
    #     episode_values = []
    #     for episode in range(2000):
    #         episode = df[df['episode'] == episode]
    #         state_0 = episode.iloc[[0]]
    #         value = state_0['state_value']
    #         episode_values.append(value)
    #         # mean = episode['state_value'].mean()
    #         # episode_values.append(mean)
    #     return episode_values
    # #
    # #
    # IS = get_value(v_IS)
    # Sarsa = get_value(v_S)
    #
    # plt.figure()
    # plt.plot(np.arange(len(IS)), IS, label="\u03C0*", color="red", linewidth=0.5)
    # plt.plot(np.arange(len(Sarsa)), Sarsa, label="\u03C0", color="grey", linewidth=0.5)
    # # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # plt.legend()
    # plt.title("TD Value estimate for policy evaluation")
    # plt.xlabel("episodes")
    # plt.ylabel("value estimate")
    # # plt.savefig("./value_estimate_covid.jpg")
    # plt.show()
    #
    #
