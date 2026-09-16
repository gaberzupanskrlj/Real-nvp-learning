import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy


# =======================================
# OBJECTIVE FUNCTION
# =======================================

def f2(tensor):
    x = tensor[:, 0]
    y = tensor[:, 1]

    q1 = (x + 0.5)**2 + (y + 0.5)**2
    q2 = (x - 0.5)**2 + (y + 0.5)**2 + 0.2

    return torch.minimum(q1, q2)


# fiksna projekcija 10D -> 2D
torch.manual_seed(1234)
M = torch.randn(10, 2)

# seed za model/trening
torch.manual_seed(0)


def f10(tensor10):

    # [batch, 10] @ [10, 2] -> [batch, 2]
    tensor2 = tensor10 @ M

    return f2(tensor2)


# =======================================
# DISKRETIZACIJA
# =======================================

def discretize(y):

    step = 2.0 / 49

    indices = torch.round(
        (y + 1.0) / step
    )

    indices = torch.clamp(
        indices,
        0,
        49
    )

    x = -1.0 + indices * step

    return x


# =======================================
# COUPLING LAYER
# =======================================

class CouplingLayer(nn.Module):

    def __init__(self):

        super().__init__()


        # input = prvih 5 koordinat
        # output = scale za drugih 5
        self.s_net = nn.Sequential(

            nn.Linear(5, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 5),
            nn.Tanh()
        )


        # translate mreža
        self.t_net = nn.Sequential(

            nn.Linear(5, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 5)
        )


    def forward(self, x):

        # 10D razdelimo na 5 + 5
        x1 = x[:, :5]
        x2 = x[:, 5:]


        # s in t sta odvisna od prve polovice
        s = self.s_net(x1)
        t = self.t_net(x1)


        # prvo polovico pustimo pri miru
        y1 = x1


        # drugo polovico transformiramo
        y2 = (
            x2 * torch.exp(s)
            + t
        )


        # sestavimo nazaj 10D vektor
        y = torch.cat(
            [y1, y2],
            dim=1
        )


        # zdaj imamo 5 scale vrednosti,
        # zato jih seštejemo
        log_det = s.sum(dim=1)


        return y, log_det


# =======================================
# REAL NVP
# =======================================

class RealNVP(nn.Module):

    def __init__(self, num_layers=4):

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

            y, log_det = layer(y)

            log_det_total += log_det


            # zamenjamo prvo in drugo polovico
            y = torch.cat(

                [
                    y[:, 5:],
                    y[:, :5]
                ],

                dim=1
            )


        # omejimo vseh 10 koordinat na (-1, 1)
        y = torch.tanh(y)


        tanh_log_det = torch.log(

            1 - y**2 + 1e-6

        ).sum(dim=1)


        log_det_total += tanh_log_det


        return y, log_det_total


# =======================================
# MODEL
# =======================================

model = RealNVP(
    num_layers=4
)


optimizer = torch.optim.Adam(

    model.parameters(),

    lr=1e-4
)


# =======================================
# TRAINING
# =======================================

BATCH_SIZE = 1024
EPOCHS = 2000


# fiksen evaluation batch
eval_generator = torch.Generator().manual_seed(123)


eval_r = torch.rand(

    20000,
    10,

    generator=eval_generator

) * 2 - 1


best_mean_f = float("inf")
best_state = None
best_epoch = 0


phase_best = float("inf")
bad_evals = 0


for epoch in range(EPOCHS):


    # -----------------------------------
    # 1. uniformna 10D distribucija
    # -----------------------------------

    r = torch.rand(

        BATCH_SIZE,
        10

    ) * 2 - 1


    # -----------------------------------
    # 2. RealNVP
    # -----------------------------------

    y, log_det = model(r)


    # -----------------------------------
    # 3. diskretizacija + objective
    # -----------------------------------

    with torch.no_grad():

        x = discretize(y)

        objective = f10(x)


        # ista normalizacija kot prej
        weights = (
            objective
            - objective.mean()
        )


        weights = weights / (

            objective.std()
            + 1e-8

        )


    # -----------------------------------
    # 4. loss
    # -----------------------------------

    loss = -(

        weights
        * log_det

    ).mean()


    # -----------------------------------
    # 5. backprop
    # -----------------------------------

    optimizer.zero_grad()

    loss.backward()


    torch.nn.utils.clip_grad_norm_(

        model.parameters(),

        max_norm=5.0

    )


    optimizer.step()


    # ===================================
    # EVALUATION vsakih 20 epochov
    # ===================================

    if epoch % 20 == 0:


        with torch.no_grad():


            eval_y, _ = model(eval_r)


            eval_x = discretize(
                eval_y
            )


            eval_objective = f10(
                eval_x
            )


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


            # projekcija 10D rešitev nazaj v 2D
            eval_projected = (
                eval_x @ M
            )


            projected_mean = (
                eval_projected
                .mean(dim=0)
            )


            hit_001 = (

                eval_objective < 0.01

            ).float().mean().item()


            hit_005 = (

                eval_objective < 0.05

            ).float().mean().item()


            hit_010 = (

                eval_objective < 0.10

            ).float().mean().item()


        # -----------------------------------
        # shrani najboljši model
        # -----------------------------------

        if eval_mean_f < best_mean_f:

            best_mean_f = eval_mean_f

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )


        # -----------------------------------
        # plateau
        # -----------------------------------

        if eval_mean_f < phase_best - 5e-5:

            phase_best = eval_mean_f

            bad_evals = 0

        else:

            bad_evals += 1


        # -----------------------------------
        # restore + manjši LR
        # -----------------------------------

        if bad_evals >= 3:


            old_lr = (
                optimizer
                .param_groups[0]["lr"]
            )


            if old_lr <= 1e-6 + 1e-12:

                print(
                    "\nMinimum learning rate reached."
                )

                print(
                    "No further improvement."
                )

                break


            new_lr = max(

                old_lr * 0.5,

                1e-6
            )


            # nazaj na najboljši checkpoint
            model.load_state_dict(
                best_state
            )


            # reset Adam momentuma
            optimizer.state.clear()


            # nov learning rate
            for param_group in optimizer.param_groups:

                param_group["lr"] = new_lr


            phase_best = best_mean_f

            bad_evals = 0


            print(

                f"\n--- PLATEAU "
                f"at epoch {epoch} ---"

            )


            print(

                f"Restored best model "
                f"from epoch {best_epoch}"

            )


            print(

                f"Learning rate: "
                f"{old_lr:.2e} "
                f"-> {new_lr:.2e}"

            )


            print(

                f"Best mean f: "
                f"{best_mean_f:.6f}\n"

            )


            # ponovno evaluiramo restore-an model
            with torch.no_grad():


                eval_y, _ = model(
                    eval_r
                )


                eval_x = discretize(
                    eval_y
                )


                eval_objective = f10(
                    eval_x
                )


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


                eval_projected = (
                    eval_x @ M
                )


                projected_mean = (
                    eval_projected
                    .mean(dim=0)
                )


                hit_001 = (

                    eval_objective < 0.01

                ).float().mean().item()


                hit_005 = (

                    eval_objective < 0.05

                ).float().mean().item()


                hit_010 = (

                    eval_objective < 0.10

                ).float().mean().item()


        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )


        print(

            f"Epoch {epoch:4d} | "

            f"mean f = "
            f"{eval_mean_f:.5f} | "

            f"best f = "
            f"{eval_best_f:.6f} | "

            f"f<0.01 = "
            f"{100 * hit_001:.2f}% | "

            f"f<0.05 = "
            f"{100 * hit_005:.2f}% | "

            f"proj mean = "
            f"("
            f"{projected_mean[0].item():+.3f}, "
            f"{projected_mean[1].item():+.3f}"
            f") | "

            f"lr = "
            f"{current_lr:.2e}"

        )


# =======================================
# PO TRENINGU
# =======================================

print()

print("Training finished.")

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best mean f:",
    best_mean_f
)


# vrnemo najboljši model
if best_state is not None:

    model.load_state_dict(
        best_state
    )


# =======================================
# FINALNA EVALUACIJA
# =======================================

model.eval()


with torch.no_grad():


    plot_y, _ = model(eval_r)


    plot_x = discretize(
        plot_y
    )


    plot_objective = f10(
        plot_x
    )


    plot_projected = (
        plot_x @ M
    )


    plot_mean_f = (
        plot_objective
        .mean()
        .item()
    )


    plot_best_f = (
        plot_objective
        .min()
        .item()
    )


    projected_mean = (

        plot_projected
        .mean(dim=0)

    )


    hit_001 = (

        plot_objective < 0.01

    ).float().mean().item()


    hit_005 = (

        plot_objective < 0.05

    ).float().mean().item()


    hit_010 = (

        plot_objective < 0.10

    ).float().mean().item()


print("\n=== BEST 10D MODEL ===")

print(
    "Best epoch:",
    best_epoch
)

print(
    "Mean f10:",
    plot_mean_f
)

print(
    "Best sampled f10:",
    plot_best_f
)

print(
    "Mean projected point:",
    projected_mean[0].item(),
    projected_mean[1].item()
)

print(
    "f < 0.01:",
    100 * hit_001,
    "%"
)

print(
    "f < 0.05:",
    100 * hit_005,
    "%"
)

print(
    "f < 0.10:",
    100 * hit_010,
    "%"
)


# =======================================
# GRAF PROJEKCIJE 10D -> 2D
# =======================================

plot_grid = torch.linspace(
    -4,
    4,
    250
)


X, Y = torch.meshgrid(

    plot_grid,
    plot_grid,

    indexing="ij"

)


grid_points = torch.stack(

    [
        X.flatten(),
        Y.flatten()
    ],

    dim=1

)


with torch.no_grad():

    F = f2(
        grid_points
    ).reshape(
        X.shape
    )


plt.figure(
    figsize=(10, 8)
)


contour = plt.contourf(

    X.numpy(),
    Y.numpy(),
    F.numpy(),

    levels=35

)


plt.colorbar(
    contour,
    label="f2(z1, z2)"
)


plt.scatter(

    plot_projected[:, 0].numpy(),
    plot_projected[:, 1].numpy(),

    s=8,
    alpha=0.25,

    label="10D diskretne točke po projekciji x @ M"

)


plt.scatter(

    [-0.5, 0.5],
    [-0.5, -0.5],

    marker="x",
    s=180,
    linewidths=2,

    label="minimuma f2"

)


plt.xlabel(
    "projekcija z1"
)

plt.ylabel(
    "projekcija z2"
)


plt.title(

    "10D RealNVP – projekcija diskretnih rešitev v 2D\n"

    f"best epoch = {best_epoch}, "
    f"mean f10 = {plot_mean_f:.4f}"

)


plt.legend()

plt.tight_layout()

plt.show()