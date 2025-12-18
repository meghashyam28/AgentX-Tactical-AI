import numpy as np
import matplotlib.pyplot as plt
from src.agent import LinUCBAgent
from env.env import DriftingEnvironment
from src.agent import RandomAgent

# Initialize Environment & Agents
print("--- Starting AgentX Sports Adaptation Simulation ---")
env = DriftingEnvironment(n_actions=5, context_dim=6)
agent_x = LinUCBAgent(n_actions=5, context_dim=6)
agent_rand = RandomAgent(n_actions=5)

h_x, h_r = [], []
steps = 10000

for i in range(1, steps + 1):
    # AT STEP 5000: OPPONENT SHIFTS TACTICS (Drift event)
    if i == 5000:
        print("\n[EVENT] Opponent changed defensive formation! AI Adapting...")
        env.theta_true += np.random.normal(0, 0.5, env.theta_true.shape)
        
    context = env.get_context()
    
    # AgentX Decision
    a_x = agent_x.select_action(context)
    r_x, _, _ = env.get_reward(a_x, context)
    agent_x.update(a_x, context, r_x)
    
    # Baseline Decision
    a_r = agent_rand.select_action(context)
    r_r, _, _ = env.get_reward(a_r, context)
    
    h_x.append(r_x)
    h_r.append(r_r)
    
    if i % 1000 == 0:
        print(f"Step {i} | AgentX Success Rate: {np.mean(h_x[-500:]):.2f}")

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(np.convolve(h_x, np.ones(500)/500, mode='valid'), label='AgentX (AI Coach)', color='blue')
plt.plot(np.convolve(h_r, np.ones(500)/500, mode='valid'), label='Static Playbook (Random)', color='red', linestyle='--')
plt.axvline(x=5000, color='gray', linestyle=':', label='Tactical Shift')
plt.title("AgentX: Real-Time Tactical Adaptation in Sports")
plt.xlabel("Game Possessions")
plt.ylabel("Success Probability (Moving Average)")
plt.legend()
plt.savefig('result.png')
print("\nSimulation complete. Graph saved as 'result.png'.")
plt.show()