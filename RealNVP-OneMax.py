import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


#spremeljivke

DIMENSION = 10
NUM_LAYERS = 4
HIDDEN_DIM = 64

BATCH_SIZE = 1024
EPOCHS =5000
LEARNING_RATE = 3e-4

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


torch.manual_seed(SEED)

#problem

def discretize(y):
    return (y > 0).float()


def onemax(x):
    return x.sum(dim=1)



class CouplingLayer(nn.Module):

    def __init__(self):
        super().__init__()

        half = DIMENSION // 2

        self.s_net = nn.Sequential(
            nn.Linear(half, HIDDEN_DIM),
            nn.ReLU(),

            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),

            nn.Linear(HIDDEN_DIM, half),
            nn.Tanh()
        )


        self.t_net = nn.Sequential(
            nn.Linear(half, HIDDEN_DIM),
            nn.ReLU(),

            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),

            nn.Linear(HIDDEN_DIM, half)
        )



    def forward(self,x):

        half = DIMENSION // 2

        x1 = x[:,:half]
        x2 = x[:,half:]


        s = self.s_net(x1)
        t = self.t_net(x1)


        y1 = x1
        y2 = x2 * torch.exp(s) + t


        y = torch.cat(
            [y1,y2],
            dim=1
        )


        log_det = s.sum(dim=1)


        return y, log_det



    def inverse(self,y):

        half = DIMENSION // 2

        y1 = y[:,:half]
        y2 = y[:,half:]


        s = self.s_net(y1)
        t = self.t_net(y1)


        x1 = y1
        x2 = (y2-t) * torch.exp(-s)


        x = torch.cat(
            [x1,x2],
            dim=1
        )


        log_det = -s.sum(dim=1)


        return x, log_det


#model

class RealNVP(nn.Module):

    def __init__(self, num_layers):

        super().__init__()

        self.layers = nn.ModuleList(
            [
                CouplingLayer()
                for _ in range(num_layers)
            ]
        )



    def forward(self,x):

        y = x

        log_det_total = torch.zeros(
            x.shape[0],
            device=x.device
        )


        half = DIMENSION // 2


        for layer in self.layers:

            y, log_det = layer(y)

            log_det_total += log_det


            y = torch.cat(
                [
                    y[:,half:],
                    y[:,:half]
                ],
                dim=1
            )


        return y, log_det_total



    def inverse(self,y):

        x = y

        log_det_total = torch.zeros(
            y.shape[0],
            device=y.device
        )


        half = DIMENSION // 2


        for layer in reversed(self.layers):

            x = torch.cat(
                [
                    x[:,half:],
                    x[:,:half]
                ],
                dim=1
            )


            x, log_det = layer.inverse(x)

            log_det_total += log_det


        return x, log_det_total

#traning
import copy

best_mean = -1
best_state = None

model = RealNVP(
    NUM_LAYERS
).to(DEVICE)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


history = []

LAMBDA_REG = 0.1
EXPLORATION_RATE = 0.05

best_mean = -1
best_state = None
for epoch in range(EPOCHS):

  

    if torch.rand(1).item() < EXPLORATION_RATE:

        r = torch.rand(
            BATCH_SIZE,
            DIMENSION,
            device=DEVICE
        ) * 2 - 1

    else:

        r = torch.rand(
            BATCH_SIZE,
            DIMENSION,
            device=DEVICE
        ) * 2 - 1


    y, log_det_forward = model(r)
    


    with torch.no_grad():

        x = discretize(y)

        objective = onemax(x)

        reward = objective / DIMENSION


   

    loss = (
        -(reward * log_det_forward).mean()
        + LAMBDA_REG * (log_det_forward ** 2).mean()
    )



    optimizer.zero_grad()

    loss.backward()


    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=5
    )


    optimizer.step()




    mean_f = objective.mean().item()

    best_f = objective.max().item()

    history.append(mean_f)

    if mean_f > best_mean:
       best_mean = mean_f
       best_state = copy.deepcopy(model.state_dict())


    if epoch % 100 == 0:

        print(
            f"{epoch:4d} | "
            f"mean={mean_f:.2f} | "
            f"best={best_f:.0f} | "
            f"loss={loss.item():.3f} | "
            f"logdet={log_det_forward.abs().mean().item():.3f}"
        )


model.load_state_dict(best_state)


#eval

with torch.no_grad():

    r = torch.rand(
        BATCH_SIZE,
        DIMENSION,
        device=DEVICE
    ) * 2 - 1


    y, log_det_forward = model(r)

    x = discretize(y)

    f = onemax(x)



print()

print("Bit probabilities:")
print(x.float().mean(dim=0))


print("Final mean:", f.mean().item())

print("Best:", f.max().item())


print(
    "Solutions f=10:",
    (f == 10).float().mean().item()
)


print()

print("f distribution:")


unique, counts = torch.unique(
    f,
    return_counts=True
)


for u, c in zip(unique, counts):
    print(
        u.item(),
        c.item()
    )


print()

print(
    "mean abs logdet:",
    log_det_forward.abs().mean().item()
)


print()

print(
    "history length:",
    len(history)
)

#graf

plt.plot(history)

plt.xlabel("Epoch")

plt.ylabel("Mean OneMax")

plt.grid()

plt.show()