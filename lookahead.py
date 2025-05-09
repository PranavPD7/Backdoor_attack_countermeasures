import torch

class Lookahead(torch.optim.Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Invalid slow update rate: {}".format(alpha))
        if not 1 <= k:
            raise ValueError("Invalid lookahead steps: {}".format(k))

        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_counter = 0

        # Cache the parameters for the slow weights
        self.param_groups = self.optimizer.param_groups
        for group in self.param_groups:
            group["slow_params"] = [p.clone().detach() for p in group["params"]]

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.optimizer.step(closure)  # Perform base optimizer step
        self.step_counter += 1

        # Perform Lookahead update every k steps
        if self.step_counter % self.k == 0:
            for group in self.param_groups:
                for p, q in zip(group["params"], group["slow_params"]):
                    if p.grad is None:
                        continue
                    q.data.add_(self.alpha, p.data - q.data)  # Update slow weights
                    p.data.copy_(q.data)  # Copy back to model parameters

        return loss

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "step_counter": self.step_counter}

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.step_counter = state_dict["step_counter"]