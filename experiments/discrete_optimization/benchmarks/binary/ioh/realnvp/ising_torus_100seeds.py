from common_100seed import RealNVPConfig, run_experiment

CONFIG = RealNVPConfig(
    problem_name="IsingTorus",
    output_dir="results/benchmark_100seeds/realnvp/ising_torus",
    problem_instance=1,
    num_layers=4,
    hidden_dim=64,
    batch_size=1024,
    learning_rate=1e-4,
    target=200.0,
)

if __name__ == "__main__":
    run_experiment(CONFIG)
