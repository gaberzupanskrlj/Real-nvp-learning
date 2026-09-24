import copy
import math
import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#spremeljikve
DIMENSION = 100

NUM_LAYERS = 4
HIDDEN_DIM = 128

BATCH_SIZE = 1024
EPOCHS = 5000
LEARNING_RATE = 1e-4

SEED = 42

VALIDATION_SIZE = 4096
TEST_SIZE = 16384

VALIDATE_EVERY = 50
LOG_EVERY = 50

MAX_GRAD_NORM = 5.0


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




torch.manual_seed(SEED)

#objective function

def discretize(y):
    """
    Continuous output -> binary solution
    """
    return (y > 0).float()


def onemax(x):
    """
    OneMax:
    maximize number of ones
    """
    return x.sum(dim=1)


class CouplingLayer(nn.Module):

    def __init__(self, dimension, hidden_dim):

        super().__init__()

        self.half = dimension // 2


        self.scale_net = nn.Sequential(
            nn.Linear(self.half, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, self.half),
            nn.Tanh()
        )


        self.translate_net = nn.Sequential(
            nn.Linear(self.half, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, self.half)
        )



    def forward(self, x):

        x1 = x[:, :self.half]
        x2 = x[:, self.half:]


        s = self.scale_net(x1)
        t = self.translate_net(x1)


        y1 = x1
        y2 = x2 * torch.exp(s) + t


        y = torch.cat(
            [y1, y2],
            dim=1
        )


        log_det = s.sum(dim=1)


        return y, log_det



    def inverse(self, y):

        y1 = y[:, :self.half]
        y2 = y[:, self.half:]


        s = self.scale_net(y1)
        t = self.translate_net(y1)


        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)


        x = torch.cat(
            [x1, x2],
            dim=1
        )


        log_det = -s.sum(dim=1)


        return x, log_det



#model

class RealNVP(nn.Module):

    def __init__(
        self,
        dimension,
        layers,
        hidden_dim
    ):

        super().__init__()


        self.layers = nn.ModuleList(
            [
                CouplingLayer(
                    dimension,
                    hidden_dim
                )
                for _ in range(layers)
            ]
        )


        self.half = dimension // 2



    def forward(self, x):

        log_det_total = x.new_zeros(
            x.shape[0]
        )


        z = x


        for layer in self.layers:

            z, log_det = layer(z)

            log_det_total += log_det


            # permutation
            z = torch.cat(
                [
                    z[:, self.half:],
                    z[:, :self.half]
                ],
                dim=1
            )


        return z, log_det_total



    def inverse(self, z):

        x = z

        log_det_total = z.new_zeros(
            z.shape[0]
        )


        for layer in reversed(self.layers):

            x = torch.cat(
                [
                    x[:, self.half:],
                    x[:, :self.half]
                ],
                dim=1
            )


            x, log_det = layer.inverse(x)

            log_det_total += log_det


        return x, log_det_total



# Probability model
#kako vrjeten je ta vzorec pri trenutnem modelu
#logpy​(y)=logpz​(z)+log∣detJ∣
def log_probability(model, y):

    z, log_det = model.inverse(y)


    log_pz = -0.5 * (
        z.square()
        + math.log(2 * math.pi)
    ).sum(dim=1)


    return log_pz + log_det



#eval

@torch.no_grad()
def evaluate(model, samples, name):

    y, log_det = model(samples)


    recovered, inverse_det = model.inverse(y)


    x = discretize(y)

    score = onemax(x)


    print("\n" + "="*50)
    print(name)
    print("="*50)


    print(
        "Mean objective:",
        score.mean().item()
    )

    print(
        "Best objective:",
        score.max().item()
    )

    print(
        "Optimal fraction:",
        (score == DIMENSION)
        .float()
        .mean()
        .item()
    )


    print(
        "Bit probabilities:"
    )

    print(
        x.mean(dim=0)
        .cpu()
        .numpy()
    )


    print(
        "Inverse error:",
        (recovered - samples)
        .abs()
        .max()
        .item()
    )


    print(
        "Mean |logdet|:",
        log_det.abs()
        .mean()
        .item()
    )



#traning
def train():


    model = RealNVP(
        DIMENSION,
        NUM_LAYERS,
        HIDDEN_DIM
    ).to(DEVICE)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    validation = torch.randn(
        VALIDATION_SIZE,
        DIMENSION,
        device=DEVICE
    )


    history = []

    loss_history = []


    best_score = -1
    best_model = None



    for epoch in range(EPOCHS):


      
        r = torch.randn(
            BATCH_SIZE,
            DIMENSION,
            device=DEVICE
        )


        with torch.no_grad():

            y, _ = model(r)

            reward = onemax(
                discretize(y)
            )


            baseline = (
                reward.sum() - reward
            ) / (
                BATCH_SIZE - 1
            )


            advantage = reward - baseline

        # REINFORCE objective

        log_prob = log_probability(
            model,
            y.detach()
        )


        loss = -(advantage * log_prob
        ).mean()

        optimizer.zero_grad()

        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM
        )


        optimizer.step()



        mean_reward = reward.mean().item()

        history.append(mean_reward)

        loss_history.append(loss.item())



        if epoch % LOG_EVERY == 0:

            print(
                f"{epoch:5d} | "
                f"mean={mean_reward:.4f} | "
                f"best={reward.max().item():.0f} | "
                f"loss={loss.item():.4f}"
            )



        if epoch % VALIDATE_EVERY == 0:

            with torch.no_grad():

                val_y,_ = model(validation)

                val_score = onemax(
                    discretize(val_y)
                ).mean().item()


            if val_score > best_score:

                best_score = val_score

                best_model = copy.deepcopy(
                    model.state_dict()
                )


    model.load_state_dict(best_model)


    return model, history, loss_history



# ============================================================
# Main
# ============================================================

def main():

    model, history, loss_history = train()


    test = torch.randn(
        TEST_SIZE,
        DIMENSION,
        device=DEVICE
    )


    evaluate(
        model,
        test,
        "Final evaluation"
    )



    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Mean OneMax")
    plt.grid()
    plt.show()



    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("REINFORCE loss")
    plt.grid()
    plt.show()



if __name__ == "__main__":
    main()
