import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import copy


GRID_SIZE = 50
LOW = -1.0
HIGH = 1.0

def f(tensor):
    x = tensor[:, 0]
    y = tensor[:, 1]

    q1 = (x + 0.5)**2 + (y + 0.5)**2
    q2 = (x - 0.5)**2 + (y + 0.5)**2 + 0.2

    return torch.minimum(q1, q2)

#diskretizacija
def discretize(y):
    step=(HIGH - LOW) / (GRID_SIZE-1)
    #kateremu mestu na gridu je y najblizje
    indices = torch.round((y - LOW) / step)

    #omejimo na veljavne indekse
    indices = torch.clamp(indices, 0, GRID_SIZE - 1)
    #indeksi -> vrednosti na gridu
    X=LOW + indices * step
    return X

class CouplingLayer(nn.Module):
    def __init__(self):
        super().__init__()

        # mreža za scale
        self.s_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        # mreža za translate
        self.t_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        # razdelimo 2D točko
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]

        # s in t sta odvisna samo od x1
        s = self.s_net(x1)
        t = self.t_net(x1)

        # x1 pustimo pri miru
        y1 = x1

        # x2 affine transformiramo
        y2 = x2 * torch.exp(s) + t

        # sestavimo nazaj 2D točko
        y = torch.cat([y1, y2], dim=1)


        log_det = s.squeeze(1)

        return y, log_det

class RealNVP(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()

        self.layers = nn.ModuleList([
            CouplingLayer()
            for _ in range(num_layers)
        ])

    def forward(self, x):

        y = x

        log_det_total = torch.zeros(
            x.shape[0],
            device=x.device
        )

        for layer in self.layers:

            # coupling transformacija
            y, log_det = layer(y)

            # prištejemo log determinant
            log_det_total += log_det

            # zamenjamo vlogi koordinat
            y = torch.flip(y, dims=[1])
        y = torch.tanh(y)
        tanh_log_det = torch.log(
    1 - y**2 + 1e-6
).sum(dim=1)
        log_det_total += tanh_log_det
    
        return y, log_det_total

eval_epochs = []
eval_mean_history = []
eval_best_history = []
eval_unique_history = []

torch.manual_seed(0)

model = RealNVP(num_layers=2)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

BATCH_SIZE = 1024
EPOCHS = 2000


# --------------------------------------------------
# fiksen evaluation batch
# --------------------------------------------------

eval_generator = torch.Generator().manual_seed(123)

eval_r = torch.rand(
    20000,
    2,
    generator=eval_generator
) * 2 - 1


# --------------------------------------------------
# shranjevanje najboljšega modela
# --------------------------------------------------

best_mean_f = float("inf")
best_state = None
best_epoch = 0


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

for epoch in range(EPOCHS):

    # 1. vzorci iz uniformne distribucije
    r = torch.rand(
        BATCH_SIZE,
        2
    ) * 2 - 1


    # 2. RealNVP
    y, log_det = model(r)


    # 3. diskretizacija + objective
    with torch.no_grad():

        x = discretize(y)

        objective = f(x)

        # baseline
        weights = objective - objective.mean()

        # originalna normalizacija baseline eksperimenta
        weights = weights / (
            objective.std() + 1e-8
        )


    # 4. surrogate loss
    loss = -(
        weights * log_det
    ).mean()


    # 5. backpropagation
    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=5.0
    )

    optimizer.step()


    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------

    if epoch % 20 == 0:

        with torch.no_grad():

            eval_y, eval_log_det = model(eval_r)

            eval_x = discretize(eval_y)

            eval_objective = f(eval_x)

            eval_mean_f = (
                eval_objective
                .mean()
                .item()
            )

            eval_best_f = (
                eval_objective
                .min()
                .item()
            )

            unique_x = torch.unique(
                eval_x,
                dim=0
            ).shape[0]


        # shrani najboljši checkpoint
        if eval_mean_f < best_mean_f:

            best_mean_f = eval_mean_f

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )


        print(
            f"Epoch {epoch:4d} | "
            f"eval mean f = {eval_mean_f:.4f} | "
            f"best f = {eval_best_f:.6f} | "
            f"unique x = {unique_x}"
        )


# --------------------------------------------------
# po treningu naloži najboljši model
# --------------------------------------------------

model.load_state_dict(best_state)

print()
print("Best epoch:", best_epoch)
print("Best mean f:", best_mean_f)

    # evaluation vsakih 20 epochov
    if epoch % 20 == 0:

        with torch.no_grad():
            eval_y, eval_log_det = model(eval_r)
            eval_x = discretize(eval_y)
            eval_objective = f(eval_x)
            unique_x = torch.unique(eval_x, dim=0).shape[0]
            eval_mean_f = eval_objective.mean().item()
            y_std = eval_y.std(dim=0)

        cov = torch.cov(eval_y.T)
        cov_det = torch.det(cov).item()

        mean_log_det = eval_log_det.mean().item()

        # TO MORA BITI ZNOTRAJ if epoch % 20
        if eval_mean_f < best_mean_f:
            best_mean_f = eval_mean_f
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
    f"Epoch {epoch:4d} | "
    f"eval mean f = {eval_mean_f:.4f} | "
    f"unique x = {unique_x} | "
    f"std y = ({y_std[0].item():.3f}, {y_std[1].item():.3f}) | "
    f"cov det = {cov_det:.6f} | "
    f"mean logdet = {mean_log_det:.3f}"
)
    