from common import RealNVPConfig, run_experiment


CONFIG = RealNVPConfig(
    problem_name="ConcatenatedTrap",
    dimension=100,
    num_layers=4,
    hidden_dim=64,
    batch_size=1024,
    epochs=1000,
    learning_rate=1e-4,
    seeds=(42,),
)


if __name__ == "__main__":
    run_experiment(CONFIG)
