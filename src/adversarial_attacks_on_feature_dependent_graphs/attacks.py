import torch

class BaseAttack():
    def __init__(self, device, model) -> None:
        self.device = device
        self.model = model

    def attack(self):
        raise NotImplementedError

    def l1_norm(self, gradient):
        avoid_zero_div = torch.tensor(1e-12, requires_grad=True).to(self.device)
    
        dims = tuple(range(1, len(gradient.shape)))
        
        l1_norm_manual = torch.max(avoid_zero_div, torch.sum(torch.abs(gradient), dims, keepdim=True))

        return l1_norm_manual

    def l2_norm(self, gradient):
        avoid_zero_div = torch.tensor(1e-12, requires_grad=True).to(self.device)
    
        dims = tuple(range(1, len(gradient.shape)))
        
        l2_norm_manual = torch.sqrt(torch.max(avoid_zero_div, torch.sum(gradient ** 2, dims, keepdim=True)))

        return l2_norm_manual

    def delta(self, norm, gradient):
        if norm == "1": 
            return gradient / self.l1_norm(gradient)
        elif norm == "2": 
            return gradient / self.l2_norm(gradient)
        elif norm == "inf": 
            return torch.sign(gradient)
        else:
            raise KeyError

class PGD(BaseAttack):

    def __init__(self, device, model) -> None:
        super().__init__(device, model)
        

    def attack(self, x: torch.Tensor, y: torch.Tensor, epsilon: float, norm: str = "2",
                         loss_fn=torch.nn.functional.cross_entropy, iterations=5):

        for _ in range(iterations):

            logits = self.model(x)
            loss = loss_fn(logits, y)

            gradient = torch.autograd.grad(loss, x)[0]

            # x_pert = torch.clamp(x + epsilon * delta[norm], 0, 1)
            x = x + epsilon * self.delta(norm, gradient)

        return x.detach()

def fast_gradient_attack(logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor, epsilon: float, norm: str = "2",
                         loss_fn=torch.nn.functional.cross_entropy):
    norm = str(norm)
    assert norm in ["1", "2", "inf"]

    ##########################################################
    # YOUR CODE HERE

    loss = loss_fn(logits, y)

    gradient = torch.autograd.grad(loss, x)[0]

    avoid_zero_div = torch.tensor(1e-12, requires_grad=True).to(y.device)
    
    dims = tuple(range(1, len(gradient.shape)))
    
    l1_norm_manual = torch.max(avoid_zero_div, torch.sum(torch.abs(gradient), dims, keepdim=True))
    
    l2_norm_manual = torch.sqrt(torch.max(avoid_zero_div, torch.sum(gradient ** 2, dims, keepdim=True)))


    delta = {
        "1": gradient / l1_norm_manual,
        "2": gradient / l2_norm_manual,
        "inf": torch.sign(gradient)
    }

    x_pert = torch.clamp(x + epsilon * delta[norm], 0, 1)

    ##########################################################

    return x_pert.detach()