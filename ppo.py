import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from ActorCritic import MLP, ActorCritic

#Define the environment we're using, setting a seed for numpy and torch
train_env = gym.make('LunarLander-v2')
test_env = gym.make('LunarLander-v2')
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)


#Define some variables to store shape info for our agent
INPUT_DIM = train_env.observation_space.shape[0]
HIDDEN_DIM = 128
OUTPUT_DIM = train_env.action_space.n

#Initialize the actor/critic MLPs and use them to create the policy agent
actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)
policy = ActorCritic(actor, critic)
torch.save(policy.state_dict(), './ppo_base.pt')

#Define a function to iterate of the agents layers and initialize them
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

policy.apply(init_weights)


#Define the learning rate and optimizer using the agents parameters
LEARNING_RATE = 0.0005
optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)



def train(env, policy, optimizer, discount_factor, ppo_steps, ppo_clip):
    ''''''

    policy.train() 
    states = []
    actions = []
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0

    state, _ = env.reset()

    while not done:

        state = torch.FloatTensor(state).unsqueeze(0)

        #append state here, not after we get the next state from env.step()
        states.append(state)
        
        action_pred, value_pred = policy(state)
                
        action_prob = F.softmax(action_pred, dim = -1)
                
        dist = distributions.Categorical(action_prob)
        
        action = dist.sample()
        
        log_prob_action = dist.log_prob(action)
        
        state, reward, terminated, truncated, _ = env.step(action.item())

        if terminated or truncated:
            done = True

        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        
        episode_reward += reward
    
    states = torch.cat(states)
    actions = torch.cat(actions)    
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)
    
    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)
    policy_loss, value_loss = update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip)

    return policy_loss, value_loss, episode_reward



def calculate_returns(rewards, discount_factor, normalize = True):
    '''Helper function to calculate intermediate returns using the rewards obtained during a rollout'''

    returns = []
    R = 0
    
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    return returns



def calculate_advantages(returns, values, normalize = True):
    '''Helper function to calculate advantage targets'''
    advantages = returns - values
    
    if normalize:    
        advantages = (advantages - advantages.mean()) / advantages.std()
        
    return advantages



def update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip):
    '''Update the actor and critic networks using the clipped surrogate objective function (EQ 7) from PPO paper: https://arxiv.org/abs/1707.063478 '''
    total_policy_loss = 0 
    total_value_loss = 0
    
    states = states.detach()
    actions = actions.detach()
    log_prob_actions = log_prob_actions.detach()
    advantages = advantages.detach()
    returns = returns.detach()
    
    #Performing ppo_steps of updates
    for _ in range(ppo_steps):
                
        #get new log prob of actions for all input states
        action_pred, value_pred = policy(states)
        value_pred = value_pred.squeeze(-1)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        
        #new log prob using old actions
        new_log_prob_actions = dist.log_prob(actions)
        
        #quotient of probs proportional to difference of logs probs (which is also safer)
        policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
        
        #calculate clamped and unclamped losses
        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - ppo_clip, max = 1.0 + ppo_clip) * advantages
        
        #use the one that is more pessimistic of the final objective
        policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()
        
        #calculate loss for the critic to improve the estimates
        value_loss = F.smooth_l1_loss(returns, value_pred).mean()
    
        #update parameters of the agents networks
        optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        optimizer.step()
    
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
    
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps



def evaluate(env, policy):
    '''Given a environment, run the agent through the environment evaluating its performance'''
    
    #put the agent in evaluation model to disable various torch features (i.e. dropout)
    policy.eval()
    

    rewards = []
    done = False
    episode_reward = 0
    state, _ = env.reset()

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            #Get the action distribution output by the agent, but trash the value since we don't need it
            action_pred, _ = policy(state)
            action_prob = F.softmax(action_pred, dim = -1)
                
        #choose the most likely action from the distribution
        action = torch.argmax(action_prob, dim = -1)
                
        #advance the environment using that action 
        state, reward, terminated, truncated, _ = env.step(action.item())

        episode_reward += reward

        #exit once the environment singles it's finished
        if terminated or truncated:
            done = True
        
    return episode_reward


#Setup some hyperparameters for training
MAX_EPISODES = 1_000 #max number of training/evaluation steps to do
DISCOUNT_FACTOR = 0.99 #amount to discount future rewards, gamma 
N_TRIALS = 25 #number of trails to average evaluation results over 
REWARD_THRESHOLD = 220 #average episode reward the agent should be able to achieve to be considered trained
PRINT_EVERY = 10 #number of train/eval steps to print after
PPO_STEPS = 5 #number of loops the agent is trained for
PPO_CLIP = 0.2 #amount the difference in action distributions should be clipped, epsilon in EQ 7 from https://arxiv.org/abs/1707.06347
SAVED_PARTIAL = False

train_rewards = []
test_rewards = []
for episode in range(1, MAX_EPISODES+1):
    
    policy_loss, value_loss, train_reward = train(train_env, policy, optimizer, DISCOUNT_FACTOR, PPO_STEPS, PPO_CLIP)
    
    test_reward = evaluate(test_env, policy)
    
    train_rewards.append(train_reward)
    test_rewards.append(test_reward)
    
    mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
    mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
    
    if episode % PRINT_EVERY == 0:
        
        print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f} | Mean Test Rewards: {mean_test_rewards:7.1f} |')
    
    if mean_test_rewards >= REWARD_THRESHOLD:
        
        print(f'Reached reward threshold in {episode} episodes')
        torch.save(policy.state_dict(), './ppo_trained.pt')
        
        break
    elif mean_test_rewards >= REWARD_THRESHOLD / 2 and not SAVED_PARTIAL:

        print(f'Partial threshold reached in {episode} episodes, saving partially trained agent')
        torch.save(policy.state_dict(), './ppo_partial.pt')
        SAVED_PARTIAL = True


#Using the history of training and testing rewards plot a graph showing the agents history
plt.figure(figsize=(12,8))
plt.plot(test_rewards, label='Test Reward')
plt.plot(train_rewards, label='Train Reward')
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Reward', fontsize=20)
plt.hlines(200, 0, len(test_rewards), color='r')
plt.legend(loc='lower right')
plt.grid()
plt.savefig('results.png')