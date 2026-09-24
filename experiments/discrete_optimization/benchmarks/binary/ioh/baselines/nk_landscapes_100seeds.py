from common_ea_100seed import EAConfig, run_experiment

CONFIG = EAConfig(
    problem_name="NKLandscapes",
    output_dir="results/benchmark_100seeds/one_plus_one_ea/nk_landscapes",
    problem_instance=1,
    target=None,
)

if __name__ == "__main__":
    run_experiment(CONFIG)
