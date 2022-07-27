import numpy as np
import pandas as pd

from env_Sarsa_IS import UtilityDisasterEnv
from Sarsa_IS import Sarsa_IS


"""
methods: 
    SARSA-IS: improve the policy and optimize the long term reward

objectives:
    find the optimal policy
    compare the rewards
"""


def train_REASA(max_episodes):
    """Run episodes and train the agent (robo advisors) for investors.
        params:
                max_episodes - the maximum number of episodes

        return:
                policy_df - actions
                reward_df - the rewards
                q_df - q values
                hat_p_list - estimated disaster probabilities
    """

    hat_p_list = []
    policy_df = pd.DataFrame()
    reward_df = pd.DataFrame()
    q_df = pd.DataFrame()

    # Initialized proposed disaster event probability
    # hat_p = dict({s_high: 0.9, s_low: 0.9, s_disaster: 0.9})
    hat_p = 0.5

    for episode in range(max_episodes):
        print("REASA episode", episode)

        # select the initial state
        state = environment.reset()

        # decreasing epsilon over time
        epsilon = 1 / (episode + 10)

        # learning rate is approaching to 0 when episodes run longer
        alpha = 1 / (episode + 10)

        # RL choose action based on observation
        action = model.choose_action(str(state), epsilon)

        # initial all zero eligibility trace
        model.eligibility_trace *= 0

        # step_hat_p_df = pd.DataFrame()
        step_hat_p_list = []
        step_policy_df = pd.DataFrame()
        step_q_df = pd.DataFrame()
        step_reward_df = pd.DataFrame()

        while True:
            # RL take action and get next observation and reward
            state_, reward, done = environment.step(state, action, theta, hat_p)

            # RL choose action based on next observation
            action_ = model.choose_action(str(state_), epsilon)

            # compute the imporance sampling weight
            w = model.IS_weight(str(state_), hat_p)

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            q_value = model.learn(str(state), action, reward, str(state_), action_, w, alpha)

            # update proposed disaster event probability.
            hat_p = model.update_RE_prob(str(state), action, reward, str(state_), action_, w, alpha)
            step_hat_p_list.append(hat_p)

            # memory the optimal policy to step_policy_df
            step_policy = pd.DataFrame({"episode": episode, "state": str(state), "action": action}, index=[0])
            step_policy_df = step_policy_df.append(step_policy, ignore_index=True)

            # memory the step rewards to step_reward_df
            reward_adjust = reward
            step_reward = pd.DataFrame({"episode": episode, "state": str(state), "rewards": reward_adjust}, index=[0])
            step_reward_df = step_reward_df.append(step_reward, ignore_index=True)

            # memory the q_value
            step_q = pd.DataFrame({"episode": episode, "state": str(state), "q_value": q_value}, index=[0])
            step_q_df = step_q_df.append(step_q, ignore_index=True)

            # let next state be current state in next step
            state = state_
            action = action_

            # To the last step
            if done:
                hat_p_mean = sum(step_hat_p_list) / len(step_hat_p_list)
                hat_p_list.append(hat_p_mean)
                break

        # memory policy episode by episode
        policy_df = pd.concat([policy_df, step_policy_df])

        # memory rewards episode by episode
        reward_df = pd.concat([reward_df, step_reward_df])

        # memory q episode by episode
        q_df = pd.concat([q_df, step_q_df])

    return policy_df, reward_df, q_df, hat_p_list


if __name__ == '__main__':
    path = "/Users/liangjiawen/PycharmProjects/TDlearning/preprocessor/covid_data.csv"
    # path = "/Users/liangjiawen/PycharmProjects/TDlearning/preprocessor/financial_crisis_data.csv"
    df = pd.read_csv(path)

    # investor type: general
    action_space = np.arange(1, 101) / 100
    # risk preference, take the mean {2.2, 2.3, ... , 8.4}
    theta = np.mean(np.arange(22, 84) / 10)

    # True rare-event probabilites
    s_high = str(np.array([0.0125, 0.05,  0.002]))
    s_low = str(np.array([0.005, 0.03, 0.002]))
    s_disater = str(np.array([-0.0275, 0.1352, 0.002])) #covid
    # s_disaster = str(np.array([-0.03, 0.06, 0.00031])) #financial crisis

    # p = dict({s_high: 0.01, s_low: 0.01, s_disaster: 0.1})
    p = 0.1

    environment = UtilityDisasterEnv(df)

    model = Sarsa_IS(df, action_space, p)
    # # #
    policy_df, reward_df, q_df, hat_p_list = train_REASA(500)
    # covid
    policy_df.to_pickle('Sarsa_IS_policy_covid.pkl')
    reward_df.to_pickle('Sarsa_IS_rewards_covid.pkl')
    q_df.to_pickle('Sarsa_IS_q_covid.pkl')
    np.save('Sarsa_IS_hat_p_covid.npy', hat_p_list)

    #
    #
    # hat_p_IS_list = np.load('Sarsa_IS_hat_p_covid.npy', allow_pickle=True)
    # reward_df = pd.read_pickle('Sarsa_IS_rewards_covid.pkl')
    # policy_df = pd.read_pickle('Sarsa_IS_policy_covid.pkl')
    #
    # # D_sampling_parameter = hat_p_df[s_disaster]#[:65000]
    # IS_sampling_parameter = hat_p_IS_list
    # plt.figure()
    # plt.plot(np.arange(len(IS_sampling_parameter)), IS_sampling_parameter, label="SARSA-IS", color="red", linewidth=0.5)
    # # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # y_major_locator = MultipleLocator(0.1)
    # ax = plt.gca()
    # ax.yaxis.set_major_locator(y_major_locator)
    # plt.legend()
    # plt.title("Sampling parameter (\u03B5\u0302)")
    # plt.xlabel("Episodes")
    # plt.ylabel("\u03B5\u0302")
    # # plt.savefig("Sampling_parameter_IS_covid.jpg")
    # plt.show()
    #
    #
    #
    # def get_reward(df):
    #     rewards = []
    #     for episode in range(500):
    #         every_episode = df[df['episode'] == episode]
    #         # state_0 = every_episode.iloc[[0]]
    #         # print(state_0)
    #         # value = state_0['rewards']
    #         value = every_episode['rewards'].mean() * 12
    #         rewards.append(value)
    #     return rewards  # list
    # episode_rewards = get_reward(reward_df)
    #
    # plt.figure()
    # plt.plot(np.arange(len(episode_rewards)), episode_rewards, label="Sarsa-IS", color="red", linewidth=0.5)
    # plt.legend()
    # plt.title("reward estimate")
    # plt.xlabel("episode")
    # plt.ylabel("reward estimate")
    # # plt.savefig("./episode_rewards_IS_covid.jpg")
    # plt.show()
    #
    # # policy
    # crisis_s_D = str(np.array([-0.03, 0.06, 0.00031]))
    # Covid_s_D = str(np.array([-0.0275, 0.1352, 0.0001]))
    #
    # #
    # # print('crisis_IS_policy_df', policy_df[18670:18750])
    # # crisis_IS_disaster_df = policy_df[policy_df["state"] == crisis_s_D]
    # # print('crisis_SARSA-IS disaster policy', crisis_IS_disaster_df)
    #
    # print('covid_IS_policy_df', policy_df[14950:15000])
    # covid_IS_disaster_df = policy_df[policy_df["state"] == Covid_s_D]
    # covid_IS_disaster_df = covid_IS_disaster_df[covid_IS_disaster_df['episode'] == 200]
    # print('covid_SARSA-IS disaster policy', covid_IS_disaster_df)