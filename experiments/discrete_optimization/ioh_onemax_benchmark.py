import ioh
import numpy as np
import matplotlib.pyplot as plt


def create_problem():
    return ioh.get_problem(
        "OneMax",
        instance=1,
        dimension=100,
        problem_class=ioh.ProblemClass.PBO
    )


def random_search(problem, budget):
    best_f = -np.inf
    history = []

    for _ in range(budget):
        x = np.random.randint(0, 2, size=100)

        f = problem(x)

        if f > best_f:
            best_f = f

        history.append(best_f)

    return np.array(history)



def evolutionary_algorithm(problem, budget):
    x = np.random.randint(0, 2, size=100)

    f = problem(x)

    best_f = f
    history = [best_f]

    for _ in range(budget - 1):

        child = x.copy()

        index = np.random.randint(0, 100)
        child[index] = 1 - child[index]

        child_f = problem(child)

        if child_f >= f:
            x = child
            f = child_f

        if f > best_f:
            best_f = f

        history.append(best_f)

    return np.array(history)



RUNS = 30
BUDGET = 1000


random_results = []
ea_results = []


for run in range(RUNS):

    print(f"Run {run+1}/{RUNS}")

    np.random.seed(run)

    problem = create_problem()
    random_results.append(
        random_search(problem, BUDGET)
    )


    np.random.seed(run)

    problem = create_problem()
    ea_results.append(
        evolutionary_algorithm(problem, BUDGET)
    )



random_results = np.array(random_results)
ea_results = np.array(ea_results)


random_mean = random_results.mean(axis=0)
random_std = random_results.std(axis=0)

ea_mean = ea_results.mean(axis=0)
ea_std = ea_results.std(axis=0)


print("\nFinal results")
print("----------------")
print(f"Random Search: {random_mean[-1]:.2f}")
print(f"(1+1)-EA: {ea_mean[-1]:.2f}")


evaluations = np.arange(1, BUDGET + 1)


plt.figure(figsize=(10, 6))


plt.plot(
    evaluations,
    random_mean,
    label="Random Search"
)

plt.fill_between(
    evaluations,
    random_mean - random_std,
    random_mean + random_std,
    alpha=0.2
)


plt.plot(
    evaluations,
    ea_mean,
    label="(1+1)-EA"
)

plt.fill_between(
    evaluations,
    ea_mean - ea_std,
    ea_mean + ea_std,
    alpha=0.2
)


plt.xlabel("Number of evaluations")
plt.ylabel("Best objective value f(x)")
plt.title("IOH Benchmark: OneMax(100)")
plt.legend()
plt.grid()
plt.ylim(0, 100)


plt.savefig(
    "OneMax_100_benchmark.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()