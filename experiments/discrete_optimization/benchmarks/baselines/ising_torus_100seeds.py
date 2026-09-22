from common_ea_100seed import EAConfig, run_experiment

CONFIG = EAConfig(
    problem_name="IsingTorus",
    output_dir="results/benchmark_100seeds/one_plus_one_ea/ising_torus",
    problem_instance=1,
    target=200.0,
)

if __name__ == "__main__":
    run_experiment(CONFIG)
