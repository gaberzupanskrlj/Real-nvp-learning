import math
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 2D objective function
def f2(x):
    x1 = x[:, 0]
    x2 = x[:, 1]

    q1 = (x1 + 0.5) ** 2 + (x2 + 0.5) ** 2
    q2 = (x1 - 0.5) ** 2 + (x2 + 0.5) ** 2 + 0.2
    return torch.minimum(q1, q2)

# Fixed 10D -> 2D projection
m_gen = torch.Generator().manual_seed(1234)
M = torch.randn(10, 2, generator=m_gen)

def f10(x):
    return f2(x @ M)

#diskretizacija
def discretize(y):
    step = 2.0 / 49
    idx = torch.round((y + 1.0) / step)
    idx = torch.clamp(idx, 0, 49)
    return -1.0 + idx * step

class CouplingLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.s_net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Tanh(),
        )

        self.t_net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )
#tanh ni potreben, saj t ni v eksponentu tako kot s: y=x2exp(s)+t


    def forward(self, x):
        x1, x2 = x[:, :5], x[:, 5:]

        s = self.s_net(x1)
        t = self.t_net(x1)

        y1 = x1
        y2 = x2 * torch.exp(s) + t

        y = torch.cat([y1, y2], dim=1)
        log_det = s.sum(dim=1)

        return y, log_det

    def inverse(self, y):
        y1, y2 = y[:, :5], y[:, 5:]

        s = self.s_net(y1)
        t = self.t_net(y1)

        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)

        x = torch.cat([x1, x2], dim=1)
        log_det_inverse = -s.sum(dim=1)

        return x, log_det_inverse

#model
class RealNVP(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [CouplingLayer() for _ in range(num_layers)]
        )

    def forward(self, x):
        y = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)

        for layer in self.layers:
            y, log_det = layer(y)
            log_det_total += log_det

            # zamenjamo obe 5d polovici
            y = torch.cat([y[:, 5:], y[:, :5]], dim=1)

        # potrebujemo, da omejimo rešitev na interval [-1,1]
        y = torch.tanh(y)

        tanh_log_det = torch.log(
            1.0 - y**2 + 1e-6
        ).sum(dim=1)
#seštevanje log-determinant vseh layeryov
        log_det_total += tanh_log_det
        return y, log_det_total

    def inverse(self, y):
        eps = 1e-6
        y = torch.clamp(y, -1 + eps, 1 - eps)

        #razveljavimo zadnji tanh, da ohranimo invertibilnost
        x = torch.atanh(y)

        log_det_inverse = -torch.log(
            1.0 - y**2 + eps
        ).sum(dim=1)

        # razveljavimo swape in coupling layere v obratnem vrstem redu
        for layer in reversed(self.layers):
            x = torch.cat([x[:, 5:], x[:, :5]], dim=1)
            x, log_det = layer.inverse(x)
            log_det_inverse += log_det

        return x, log_det_inverse

#priprave na training part

torch.manual_seed(0)

model = RealNVP(num_layers=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

batch_size = 1024
epochs = 2000
#fiksen evalutaion normal distribution batch
eval_gen = torch.Generator().manual_seed(123)
eval_r = torch.randn(20000, 10, generator=eval_gen)

best_mean_f = float("inf")
best_state = None
best_epoch = 0

phase_best = float("inf")
bad_evals = 0


# inverse check
with torch.no_grad():
    test_r = torch.randn(1000, 10)
    test_y, _ = model(test_r)
    test_r_back, _ = model.inverse(test_y)

    inverse_error = torch.max(
        torch.abs(test_r - test_r_back)
    ).item()

print(f"Max inverse error: {inverse_error:.3e}")

#training
# zgodovina za learning curve
history_epoch = []
history_mean_f = []
history_hit_001 = []
history_hit_005 = []
history_lr = []

for epoch in range(epochs):
    # normalna distribucija
    r = torch.randn(batch_size, 10)
#distribucija damo skozi modeol
    y, _ = model(r)

    # iz zveznih podatkov pretvorimo v diskritizirane
    with torch.no_grad():#ne pustimo modelu, da backpropagate skozi diskretizaicjo
        x = discretize(y)
        objective = f10(x)

        weights = objective - objective.mean()
        weights = weights / (objective.std() + 1e-8)

    # y naj bo obravnavan kot fiksna točka
    y_fixed = y.detach()

    r_inverse, inverse_log_det = model.inverse(y_fixed)

    log_prob_r = (
        -0.5
        * (
            r_inverse**2
            + math.log(2.0 * math.pi)
        ).sum(dim=1)
   )

    
#change of variables
    log_prob_y = log_prob_r + inverse_log_det

    loss = (weights * log_prob_y).mean()
#varnostni check
    if not torch.isfinite(loss):
        print(f"NaN/Inf loss at epoch {epoch}")
        break

    optimizer.zero_grad()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) #omejimo gradient, da ne bi postal prevelik
    optimizer.step()

#evaluation


    if epoch % 20 != 0:
        continue

    with torch.no_grad():
        eval_y, _ = model(eval_r)
        eval_x = discretize(eval_y)
        eval_objective = f10(eval_x)
        eval_projected = eval_x @ M

        eval_mean_f = eval_objective.mean().item()
        eval_best_f = eval_objective.min().item()
        projected_mean = eval_projected.mean(dim=0)

        hit_001 = (eval_objective < 0.01).float().mean().item()
        hit_005 = (eval_objective < 0.05).float().mean().item()
        hit_010 = (eval_objective < 0.10).float().mean().item()

    if eval_mean_f < best_mean_f:
        best_mean_f = eval_mean_f
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
#plateau detection
    if eval_mean_f < phase_best - 5e-5:
        phase_best = eval_mean_f
        bad_evals = 0
    else:
        bad_evals += 1

    if bad_evals >= 3:
        old_lr = optimizer.param_groups[0]["lr"]

        if old_lr <= 1e-6 + 1e-12:
            print("\nMinimum learning rate reached.")
            break
#zmanjšamo learning rate
        new_lr = max(old_lr * 0.5, 1e-6)
#resetiramo adamo momentum
        model.load_state_dict(best_state)
        optimizer.state.clear()

        for group in optimizer.param_groups:
            group["lr"] = new_lr

        phase_best = best_mean_f
        bad_evals = 0

        print(
            f"\nPlateau at epoch {epoch}: "
            f"restore epoch {best_epoch}, "
            f"lr {old_lr:.2e} -> {new_lr:.2e}"
        )

        with torch.no_grad():
            eval_y, _ = model(eval_r)
            eval_x = discretize(eval_y)
            eval_objective = f10(eval_x)
            eval_projected = eval_x @ M

            eval_mean_f = eval_objective.mean().item()
            eval_best_f = eval_objective.min().item()
            projected_mean = eval_projected.mean(dim=0)

            hit_001 = (eval_objective < 0.01).float().mean().item()
            hit_005 = (eval_objective < 0.05).float().mean().item()
            hit_010 = (eval_objective < 0.10).float().mean().item()

    current_lr = optimizer.param_groups[0]["lr"]

    history_epoch.append(epoch)
    history_mean_f.append(eval_mean_f)
    history_hit_001.append(hit_001)
    history_hit_005.append(hit_005)
    history_lr.append(current_lr)

print(
        f"Epoch {epoch:4d} | "
        f"mean f = {eval_mean_f:.5f} | "
        f"best f = {eval_best_f:.6f} | "
        f"f<0.01 = {100 * hit_001:5.2f}% | "
        f"f<0.05 = {100 * hit_005:5.2f}% | "
        f"proj mean = "
        f"({projected_mean[0]:+.3f}, {projected_mean[1]:+.3f}) | "
        f"lr = {current_lr:.2e}"
    )


print("\nTraining finished.")
print("Best epoch:", best_epoch)
print("Best mean f:", best_mean_f)

if best_state is not None:
    model.load_state_dict(best_state)
with torch.no_grad():

    test_r = torch.randn(10000, 10)

    test_y, forward_log_det = model(test_r)

    test_r_back, inverse_log_det = model.inverse(test_y)

    reconstruction_error = torch.abs(
        test_r - test_r_back
    )

    print(
        "Mean inverse error:",
        reconstruction_error.mean().item()
    )

    print(
        "Max inverse error:",
        reconstruction_error.max().item()
    )

#preverjanje invertibilnosti
    log_q_forward = (
        -0.5 * (
            test_r**2
            + math.log(2 * math.pi)
        ).sum(dim=1)
    )

    log_p_forward = (
        log_q_forward
        - forward_log_det
    )


    log_q_inverse = (
        -0.5 * (
            test_r_back**2
            + math.log(2 * math.pi)
        ).sum(dim=1)
    )

    log_p_inverse = (
        log_q_inverse
        + inverse_log_det
    )


    density_error = torch.abs(
        log_p_forward
        - log_p_inverse
    )

    print(
        "Mean log-density error:",
        density_error.mean().item()
    )

    print(
        "Max log-density error:",
        density_error.max().item()
    )

model.eval()
#graf
with torch.no_grad():
    plot_y, _ = model(eval_r)
    plot_x = discretize(plot_y)

    plot_objective = f10(plot_x)
    plot_projected = plot_x @ M

    plot_mean_f = plot_objective.mean().item()
    plot_best_f = plot_objective.min().item()
    projected_mean = plot_projected.mean(dim=0)

    hit_001 = (plot_objective < 0.01).float().mean().item()
    hit_005 = (plot_objective < 0.05).float().mean().item()
    hit_010 = (plot_objective < 0.10).float().mean().item()


print("\n=== BEST 10D GAUSSIAN MODEL ===")
print("Best epoch:", best_epoch)
print("Mean f10:", plot_mean_f)
print("Best sampled f10:", plot_best_f)
print(
    "Mean projected point:",
    projected_mean[0].item(),
    projected_mean[1].item(),
)
print("f < 0.01:", 100 * hit_001, "%")
print("f < 0.05:", 100 * hit_005, "%")
print("f < 0.10:", 100 * hit_010, "%")


# 2D view of the projected 10D solutions
grid = torch.linspace(-4, 4, 250)
X, Y = torch.meshgrid(grid, grid, indexing="ij")

points = torch.stack(
    [X.flatten(), Y.flatten()],
    dim=1,
)

with torch.no_grad():
    F = f2(points).reshape(X.shape)

plt.figure(figsize=(10, 8))

contour = plt.contourf(
    X.numpy(),
    Y.numpy(),
    F.numpy(),
    levels=35,
)

plt.colorbar(contour, label="f2(z1, z2)")

plt.scatter(
    plot_projected[:, 0].numpy(),
    plot_projected[:, 1].numpy(),
    s=8,
    alpha=0.25,
    label="10D discrete solutions after x @ M",
)

plt.scatter(
    [-0.5, 0.5],
    [-0.5, -0.5],
    marker="x",
    s=180,
    linewidths=2,
    label="f2 minima",
)

plt.xlabel("projection z1")
plt.ylabel("projection z2")

plt.title(
    "10D RealNVP with Gaussian base\n"
    f"best epoch = {best_epoch}, mean f10 = {plot_mean_f:.4f}"
)

plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(9, 5))

plt.plot(
    history_epoch,
    history_mean_f,
    marker="o",
    markersize=3,
)

plt.xlabel("Epoch")
plt.ylabel("Mean objective f10")
plt.title("Learning curve – mean objective during training")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(9, 5))

plt.plot(
    history_epoch,
    [100 * x for x in history_hit_001],
    label="f < 0.01",
)

plt.plot(
    history_epoch,
    [100 * x for x in history_hit_005],
    label="f < 0.05",
)

plt.xlabel("Epoch")
plt.ylabel("Solutions [%]")
plt.title("Learning curve – proportion of good solutions")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()