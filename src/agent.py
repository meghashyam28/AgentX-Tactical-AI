import numpy as np

class LinUCBAgent:
    def __init__(self, n_actions, context_dim, alpha=1.2):
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.alpha = alpha
        
        # Initialize A as a list of Identity Matrices (d x d)
        self.A = [np.identity(context_dim) for _ in range(n_actions)]
        # Initialize b as a list of Vectors (d,)
        self.b = [np.zeros(context_dim) for _ in range(n_actions)]

    def select_action(self, context):
        # Ensure context is a flat 1D array of size (d,)
        context = context.flatten()
        p = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            # Compute the inverse of A_a
            A_inv = np.linalg.inv(self.A[a])
            # theta = A_inv * b
            theta = A_inv @ self.b[a]
            
            # UCB formula: mean + alpha * uncertainty
            mean = theta @ context
            uncertainty = self.alpha * np.sqrt(context @ A_inv @ context)
            p[a] = mean + uncertainty
            
        return np.argmax(p)

    def update(self, action, context, reward):
        # Ensure context is flat
        context = context.flatten()
        
        # Update A: A = A + outer_product(context, context)
        # This creates a (d x d) matrix from the (d,) vector
        self.A[action] += np.outer(context, context)
        
        # Update b: b = b + reward * context
        # This adds a (d,) vector to a (d,) vector
        self.b[action] += reward * context

class RandomAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
    def select_action(self, context):
        return np.random.randint(self.n_actions)
    def update(self, *args):
        pass
