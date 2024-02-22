algorithm_name="TDS" # we insert the name of the algorithm to be registered in the CSV files to illustrations
enviroment_name='Humanoid-v3'
seed=4
start_timesteps=10000


def policy_evaluation(agent, enviroment_name,episodes=10):
    evaluation_env = gym.make(enviroment_name)
    average_reward = 0.
    for _ in range(episodes):
        state, _ = evaluation_env.reset()
        done=False
        truncuated=False
        while (not done) and (not truncuated):
            _,action = agent.choose_action(np.array(state))
            state, reward, done, truncuated,_ = evaluation_env.step(action)
            average_reward += reward
    average_reward /= episodes
    return average_reward
