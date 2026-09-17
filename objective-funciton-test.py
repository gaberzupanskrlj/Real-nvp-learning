import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy


GRID_SIZE = 50
LOW = -1.0
HIGH = 1.0

# OBJECTIVE FUNCITION

def f(tensor):
    x = tensor[:, 0]
    y = tensor[:, 1]

    q1 = (x + 0.5) ** 2 + (y + 0.5) ** 2
    q2 = (x - 0.5) ** 2 + (y + 0.5) ** 2 + 0.2

    return torch.minimum(q1, q2)



# DISKRETIZACIJA

def discretize(y):
    step = (HIGH - LOW) / (GRID_SIZE - 1)

    # kateremu mestu na gridu je y najbližje
    indices = torch.round((y - LOW) / step)

    # omejimo na veljavne indekse
    indices = torch.clamp(indices, 0, GRID_SIZE - 1)

    # indeksi -> vrednosti na gridu
    x = LOW + indices * step

    return x



# COUPLING LAYER

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
            nn.Tanh(),
        )

        # mreža za translate
        self.t_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
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

        # log |det J| za affine coupling
        log_det = s.squeeze(1)

        return y, log_det


# REAL NVP

class RealNVP(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()

        self.layers = nn.ModuleList(
            [CouplingLayer() for _ in range(num_layers)]
        )

    def forward(self, x):
        y = x

        log_det_total = torch.zeros(
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )

        for layer in self.layers:
            # coupling transformacija
            y, log_det = layer(y)

            # prištejemo log determinant
            log_det_total += log_det

            # zamenjamo vlogi koordinat
            y = torch.flip(y, dims=[1])

        # omejimo output na (-1, 1)
        y = torch.tanh(y)

        # Jacobian tanh transformacije:
        # d/dx tanh(x) = 1 - tanh(x)^2
        tanh_log_det = torch.log(
            1.0 - y**2 + 1e-6
        ).sum(dim=1)

        log_det_total += tanh_log_det

        return y, log_det_total



# SETUP

torch.manual_seed(0)

model = RealNVP(num_layers=4)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    threshold=1e-4,
    min_lr=1e-6,
)

BATCH_SIZE = 1024
EPOCHS = 2000

# fiksen eval batch
eval_generator = torch.Generator().manual_seed(123)

eval_r = torch.rand(
    20000,
    2,
    generator=eval_generator,
) * 2 - 1



# CHECKPOINT / PHASE SETUP

best_mean_f = float("inf")
best_state = None
best_epoch = 0

normalized_phase = True

phase_best = float("inf")
bad_evals = 0

MIN_DELTA = 5e-4
SWITCH_PATIENCE = 2

switch_epoch = None



# TRAINING

for epoch in range(EPOCHS):

    
    # 1. vzorci iz uniformne distribucije
   
    r = torch.rand(BATCH_SIZE, 2) * 2 - 1


    # 2. RealNVP
   
    y, log_det = model(r)

   
    # 3. diskretizacija + objective
 
    with torch.no_grad():
        x = discretize(y)
        objective = f(x)

        # FAZA 1:
        # centrirani + standardizirani weights
        if normalized_phase:
            weights = (
                objective - objective.mean()
            ) / (objective.std() + 1e-8)

        # FAZA 2:
        # fine tuning samo s centriranjem
        else:
            weights = objective - objective.mean()

   
    # 4. loss
   
    loss = -(weights * log_det).mean()

   
    # 5. backpropagation
   
    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=5.0,
    )

    optimizer.step()

 
    # EVALUATION vsakih 20 epochov
   
    if epoch % 20 == 0:

        with torch.no_grad():
            eval_y, _ = model(eval_r)
            eval_x = discretize(eval_y)
            eval_objective = f(eval_x)

            eval_mean_f = eval_objective.mean().item()
            eval_best_f = eval_objective.min().item()

            unique_x = torch.unique(
                eval_x,
                dim=0,
            ).shape[0]

      

        if eval_mean_f < best_mean_f:
            best_mean_f = eval_mean_f
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

      
        just_switched = False

        if normalized_phase:

            if eval_mean_f < phase_best - MIN_DELTA:
                phase_best = eval_mean_f
                bad_evals = 0

            else:
                bad_evals += 1

            # plateau -> preklop
            if bad_evals >= SWITCH_PATIENCE:
                normalized_phase = False
                just_switched = True
                switch_epoch = epoch

                # vrnemo se na najboljši checkpoint
                if best_state is not None:
                    model.load_state_dict(best_state)

                # reset Adam momentuma
                optimizer.state.clear()

                # manjši learning rate
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 5e-5

                bad_evals = 0

                # ponovno evaluiramo RESTORAN model,
                # da izpis in scheduler uporabljata pravo stanje
                with torch.no_grad():
                    eval_y, _ = model(eval_r)
                    eval_x = discretize(eval_y)
                    eval_objective = f(eval_x)

                    eval_mean_f = eval_objective.mean().item()
                    eval_best_f = eval_objective.min().item()

                    unique_x = torch.unique(
                        eval_x,
                        dim=0,
                    ).shape[0]

        # -----------------------------------
        # scheduler samo po preklopu
        # -----------------------------------
        if not normalized_phase and not just_switched:
            scheduler.step(eval_mean_f)

     
     
        current_lr = optimizer.param_groups[0]["lr"]

   
        if normalized_phase:
            phase = "NORMALIZED"
        else:
            phase = "FINE-TUNE"

        
        
        print(
            f"Epoch {epoch:4d} | "
            f"eval mean f = {eval_mean_f:.4f} | "
            f"best f = {eval_best_f:.6f} | "
            f"unique x = {unique_x} | "
            f"phase = {phase} | "
            f"lr = {current_lr:.2e}"
        )


# =======================================
# PO KONCU TRENINGA
# =======================================
print()
print("Training finished.")
print("Best epoch:", best_epoch)
print("Best mean f:", best_mean_f)
print("Switch epoch:", switch_epoch)

if best_state is not None:
    model.load_state_dict(best_state)


# =======================================
# GRAF NAJBOLJŠEGA MODELA
# =======================================
model.eval()

# ponovno izračunamo output najboljšega modela
with torch.no_grad():
    plot_y, _ = model(eval_r)

    # diskretizirane točke
    plot_x = discretize(plot_y)

    plot_objective = f(plot_x)

    plot_mean_f = plot_objective.mean().item()
    plot_best_f = plot_objective.min().item()


print("\n=== BEST MODEL ===")
print("Best epoch:", best_epoch)
print("Mean f:", plot_mean_f)
print("Best f:", plot_best_f)


# ---------------------------------------
# naredimo gosto mrežo samo za ozadje
# ---------------------------------------
plot_grid = torch.linspace(
    LOW,
    HIGH,
    200,
)

X, Y = torch.meshgrid(
    plot_grid,
    plot_grid,
    indexing="ij",
)

grid_points = torch.stack(
    [
        X.flatten(),
        Y.flatten(),
    ],
    dim=1,
)

with torch.no_grad():
    F = f(
        grid_points
    ).reshape(
        X.shape
    )


# =======================================
# RISANJE
# =======================================
plt.figure(
    figsize=(9, 7)
)

contour = plt.contourf(
    X.numpy(),
    Y.numpy(),
    F.numpy(),
    levels=30,
)

plt.colorbar(
    contour,
    label="f(x, y)",
)


# ---------------------------------------
# gostota diskretnih točk
# ---------------------------------------
unique_points, counts = torch.unique(
    plot_x,
    dim=0,
    return_counts=True,
)

# več zadetkov -> večja pika
sizes = 10 + 250 * (
    counts.float() / counts.max()
)

plt.scatter(
    unique_points[:, 0].numpy(),
    unique_points[:, 1].numpy(),
    s=sizes.numpy(),
    alpha=0.6,
    label="diskretne točke (velikost = pogostost)",
)

# minimuma funkcije
plt.scatter(
    [-0.5, 0.5],
    [-0.5, -0.5],
    marker="x",
    s=180,
    linewidths=2,
    label="minimuma funkcije",
)

plt.xlabel("x")
plt.ylabel("y")

plt.title(
    "Diskretizirane točke čez funkcijo f\n"
    f"best epoch = {best_epoch}, "
    f"mean f = {plot_mean_f:.4f}"
)

plt.xlim(
    LOW,
    HIGH,
)

plt.ylim(
    LOW,
    HIGH,
)

plt.legend()
plt.tight_layout()
plt.show()


# =======================================
# DODATNE METRIKE
# =======================================
mean_x = plot_x[:, 0].mean().item()
mean_y = plot_x[:, 1].mean().item()

hit_001 = (plot_objective < 0.01).float().mean().item()
hit_005 = (plot_objective < 0.05).float().mean().item()
hit_010 = (plot_objective < 0.10).float().mean().item()

print("Mean point:", mean_x, mean_y)


# =======================================
# DELEŽ TOČK NA GLOBALNEM GRID OPTIMUMU
# =======================================
grid_values = torch.linspace(
    LOW,
    HIGH,
    GRID_SIZE,
)

GX, GY = torch.meshgrid(
    grid_values,
    grid_values,
    indexing="ij",
)

all_grid_points = torch.stack(
    [GX.flatten(), GY.flatten()],
    dim=1,
)

with torch.no_grad():
    all_grid_f = f(all_grid_points)
    grid_min_f = all_grid_f.min()


optimal_prob = torch.isclose(
    plot_objective,
    grid_min_f,
    atol=1e-8,
    rtol=0.0,
).float().mean().item()

print("Grid optimum f:", grid_min_f.item())
print("P(optimal grid point):", 100 * optimal_prob, "%")
print("f < 0.01:", 100 * hit_001, "%")
print("f < 0.05:", 100 * hit_005, "%")
print("f < 0.10:", 100 * hit_010, "%")
