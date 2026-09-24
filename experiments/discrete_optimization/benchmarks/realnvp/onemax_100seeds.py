from common_100seed import RealNVPConfig, run_experiment

CONFIG = RealNVPConfig(
    problem_name="OneMax",
    output_dir="results/benchmark_100seeds/realnvp/onemax",
    num_layers=4,
    hidden_dim=128,
    batch_size=1024,
    learning_rate=1e-4,
    target=100.0,
    n_seeds=100,
    workers=24,
)

if __name__ == "__main__":
    run_experiment(CONFIG)
