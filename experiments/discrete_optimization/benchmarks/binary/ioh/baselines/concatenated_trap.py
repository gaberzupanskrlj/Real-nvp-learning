from one_plus_one_ea import EAConfig, run_experiment


CONFIG = EAConfig(
    problem_name="ConcatenatedTrap",
    dimension=100,
    max_evaluations=4_096_000,
    seeds=tuple(range(42, 52)),
    target=20.0,
)


if __name__ == "__main__":
    run_experiment(CONFIG)
