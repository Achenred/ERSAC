from risk_agent_utils import ModelFreeRiskRL, GridWorldMDP, GridWorldEnv, ProspectAgent, GridDisplay

grid_rows = 4
grid_cols = 4
num_actions = 4
terminal_states = [0, 15]


mdp = GridWorldMDP(grid_rows, grid_cols, num_actions, terminal_states)

env = GridWorldEnv(grid_rows, grid_cols, num_actions, terminal_states)


n = env.n
m = env.m
states = env.states
actions = env.actions


model_free_risk_rl = ModelFreeRiskRL(n,m,states,actions)
agent = ProspectAgent(rho_plus=.01)

model_free_risk_rl.risk_q_learning(env, agent)


display = GridDisplay(model_free_risk_rl, env)
display.show_values(title='Grid World: Risk Sensitive Q-Learning Values')
display.show_q_values(title='Grid World: Risk Sensitive Q-Learning Q-Values')

model_free_risk_rl.eu_q_learning(env, agent)

display = GridDisplay(model_free_risk_rl, env)
display.show_values(title='Grid World: Expected Utility Q-Learning Values')
display.show_q_values(title='Grid World: Expected Utiltiy Q-Learning Q-Values')

# import gym
#
# env = "CartPole-v0"
# env = gym.make(env)
#
# model_free_risk_rl = ModelFreeRiskRL()
# agent = ProspectAgent(rho_plus=.01)
#
# model_free_risk_rl.risk_q_learning(env, agent)



