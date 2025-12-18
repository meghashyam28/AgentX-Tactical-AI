import numpy as np

class DriftingEnvironment:
    """
    Simulates a dynamic sports environment where 'optimal' tactics
    slowly change over time due to opponent adaptation or fatigue.
    """
    def __init__(self, n_actions=5, context_dim=6, drift_rate=0.01):
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.drift_rate = drift_rate
        
        # Hidden 'truth': The relationship between context and success
        # This is what the AI tries to learn.
        self.theta_true = np.random.randn(n_actions, context_dim)

    def get_context(self):
        """
        Generates the current 'State' of the game.
        Features: [Score Gap, Time Left, Fatigue, Crowd Noise, Pressure, Field Position]
        """
        return np.random.uniform(-1, 1, self.context_dim)

    def get_reward(self, action, context):
        """
        Calculates if the play was successful (1) or not (0).
        """
        # Linear relationship: Probability = context dot-product with truth
        # Plus some noise to make it realistic
        prob = 1 / (1 + np.exp(-(context @ self.theta_true[action])))
        reward = 1 if np.random.rand() < prob else 0
        
        # SLOW DRIFT: The world changes slightly every play
        self.theta_true += np.random.normal(0, self.drift_rate, self.theta_true.shape)
        
        return reward, prob, None
