algorithm_name="TDS" # we insert the name of the algorithm to be registered in the CSV files to illustrations
enviroment_name='Humanoid-v3'
seed=4

timestep_limit=500000
evaluationStep=5000
printStep=50
initial_timestep=10000

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
    
env = gym.make(enviroment_name)
env.action_space.seed(seed)
#torch.manual_seed(seed) #set Torch or Tensorflow seed if used
#numpy.random.seed(seed) # set Numpy seed if used

agent = Agent(#)
    
evaluations = [policy_evaluation(agent,enviroment_name)]

average_rewards=[]
total_rewards=[]

steps=0
while (steps <= timestep_limit):
    state, _ = env.reset(seed=seed)
    done, truncuated = False, False
    while (not done) and (not truncuated):
        if steps < initial_timestep:
            action = env.action_space.sample()
        else:
            action = (policy.select_action(np.array(state))+ np.random.normal(0, max_action * Beta, size=action_dim)).clip(-max_action, max_action)
        state_, reward, done, truncuated, info = env.step(action)
        replay_buffer.add(state, action, state_, reward, done)
        steps += 1
        state = state_
        if steps>=initial_timestep:
            policy.train(replay_buffer, 100)
        if(steps%5000)==0:
            evaluation_reward = eval_policy(policy, envName)
            evaluations.append(evaluation_reward)
            print(f"Evaluation over {10} episodes: {evaluation_reward:.3f}  step{steps}")


variant = dict(algorithm=algorithm_name,env=enviroment_name,)
if not os.path.exists(f"./data/{enviroment_name}/{algorithm_name}/seed{seed}"):
    os.makedirs(f'./data/{enviroment_name}/{algorithm_name}/seed{seed}')
with open(f'./data/{enviroment_name}/{algorithm_name}/seed{seed}/variant.json', 'w') as outfile:
    json.dump(variant,outfile)
data = np.array(evaluations)
df = pd.DataFrame(data=data,columns=["Average Return"]).reset_index()
df['Timesteps'] = df['index'] * 5000
df['env'] = enviroment_name
df['algorithm_name'] = algorithm_name
df.to_csv(f'./data/{enviroment_name}/{algorithm_name}/seed{seed}/progress.csv', index = False)
