from common import RealNVPConfig, run_experiment


CONFIG = RealNVPConfig(
    problem_name="LeadingOnes",
    dimension=100,
    num_layers=8,
    hidden_dim=150,
    batch_size=4096,
    epochs=1000,
    learning_rate=1e-4,
    seeds=(42,),
)


if __name__ == "__main__":
    run_experiment(CONFIG)
