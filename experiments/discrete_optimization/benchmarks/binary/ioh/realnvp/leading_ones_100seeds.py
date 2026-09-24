from common_100seed import RealNVPConfig, run_experiment

CONFIG = RealNVPConfig(
    problem_name="LeadingOnes",
    output_dir="results/benchmark_100seeds/realnvp/leading_ones",
    num_layers=8,
    hidden_dim=150,
    batch_size=4096,
    learning_rate=1e-4,
    target=100.0,
)

if __name__ == "__main__":
    run_experiment(CONFIG)
