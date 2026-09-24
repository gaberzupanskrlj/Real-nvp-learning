import matplotlib.pyplot as plt

# Current exploratory benchmark results.
# Higher is better on every objective shown here.

known = {
    "OneMax": {"max": 100, "RealNVP": 100, "(1+1)-EA": 100},
    "LeadingOnes": {"max": 100, "RealNVP": 97, "(1+1)-EA": 100},
    "ConcatenatedTrap": {"max": 20, "RealNVP": 16, "(1+1)-EA": 16.2},
    "IsingTorus": {"max": 200, "RealNVP": 180, "(1+1)-EA": 200},
}

nk = {
    "RealNVP": (-0.3046623965, 0.0039148876),
    "(1+1)-EA": (-0.2926634223, 0.0004711073),
}

ising = {
    "RealNVP": (173.3333333333, 7.0237691686),
    "(1+1)-EA": (194.0, 9.6609178308),
}

# 1) Normalized overview
problems = list(known)
realnvp = [100 * known[p]["RealNVP"] / known[p]["max"] for p in problems]
ea = [100 * known[p]["(1+1)-EA"] / known[p]["max"] for p in problems]
y = list(range(len(problems)))

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(realnvp, [v + 0.10 for v in y], s=90, label="RealNVP")
ax.scatter(ea, [v - 0.10 for v in y], s=90, marker="s", label="(1+1)-EA")
for yi, r, e in zip(y, realnvp, ea):
    ax.plot([r, e], [yi + 0.10, yi - 0.10], linewidth=1.4, alpha=0.55)
    ax.annotate(f"{r:.1f}%", (r, yi + 0.10), xytext=(7, 0), textcoords="offset points", va="center")
    ax.annotate(f"{e:.1f}%", (e, yi - 0.10), xytext=(7, 0), textcoords="offset points", va="center")
ax.axvline(100, linestyle="--", linewidth=1.2)
ax.set_yticks(y)
ax.set_yticklabels(problems)
ax.invert_yaxis()
ax.set_xlim(75, 103)
ax.set_xlabel("Best observed performance (% of known optimum)")
ax.set_title("RealNVP vs (1+1)-EA")
ax.legend(frameon=False)
ax.grid(axis="x", alpha=0.18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("benchmark_overview.png", dpi=240, bbox_inches="tight")
plt.show()

# 2) NKLandscapes mean best +- SD
labels = list(nk)
means = [nk[a][0] for a in labels]
stds = [nk[a][1] for a in labels]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.errorbar(range(len(labels)), means, yerr=stds, fmt="o", markersize=9, capsize=7, linewidth=1.8)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_ylabel("Mean best objective")
ax.set_title("NKLandscapes 100D — mean best ± SD")
ax.grid(axis="y", alpha=0.18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("nk_landscapes.png", dpi=240, bbox_inches="tight")
plt.show()

# 3) IsingTorus mean best +- SD
means = [ising[a][0] for a in labels]
stds = [ising[a][1] for a in labels]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.errorbar(range(len(labels)), means, yerr=stds, fmt="o", markersize=9, capsize=7, linewidth=1.8)
ax.axhline(200, linestyle="--", linewidth=1.2, label="Global optimum = 200")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_ylim(150, 208)
ax.set_ylabel("Best objective per run")
ax.set_title("IsingTorus 100D — mean best ± SD")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("ising_torus.png", dpi=240, bbox_inches="tight")
plt.show()
