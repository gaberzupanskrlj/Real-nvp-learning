from common_ea_100seed import EAConfig, run_experiment

CONFIG = EAConfig(
    problem_name="OneMax",
    output_dir="results/benchmark_100seeds/one_plus_one_ea/onemax",
    target=100.0,
)

if __name__ == "__main__":
    run_experiment(CONFIG)
