from common_ea_100seed import EAConfig, run_experiment

CONFIG = EAConfig(
    problem_name="LeadingOnes",
    output_dir="results/benchmark_100seeds/one_plus_one_ea/leading_ones",
    target=100.0,
)

if __name__ == "__main__":
    run_experiment(CONFIG)
