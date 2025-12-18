import numpy as np
from src.agent import LinUCBAgent
from env.env import DriftingEnvironment

# Load components from structured paths
agent = LinUCBAgent(n_actions=5, context_dim=6, alpha=1.5)
env = DriftingEnvironment(n_actions=5, context_dim=6)

print("--- Running AgentX Autonomous Demo ---")
for i in range(10):
    context = env.get_context() #
    action = agent.select_action(context) #
    reward, _, _ = env.get_reward(action, context) #
    agent.update(action, context, reward) #
    print(f"Possession {i+1}: Play {action} | Success: {reward}")