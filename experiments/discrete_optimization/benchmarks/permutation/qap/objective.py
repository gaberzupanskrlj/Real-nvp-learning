import torch


# ============================================================
# QAPLIB Nug20 benchmark
# ============================================================

N = 20
OPTIMUM = 2570


# ============================================================
# QAPLIB optimal solution
# ============================================================
#
# QAPLIB gives the Nug20 optimal permutation as:
#
#   18 14 10 3 9 4 2 12 11 16
#   19 15 20 8 13 17 5 7 1 6
#
# in 1-based indexing.
#
# QAPLIB's solution convention is:
#
#       location -> facility
#
# while our objective / RealNVP decoder uses:
#
#       facility -> location
#
# Therefore we store the original QAPLIB permutation and
# automatically invert it.
#

QAPLIB_OPTIMAL_PERMUTATION = torch.tensor(
    [
        18, 14, 10, 3, 9,
        4, 2, 12, 11, 16,
        19, 15, 20, 8, 13,
        17, 5, 7, 1, 6,
    ],
    dtype=torch.long,
)


def invert_qaplib_permutation(
    permutation_1based: torch.Tensor,
) -> torch.Tensor:
    """
    Convert the QAPLIB convention location -> facility into our
    facility -> location convention, and 1-based into 0-based indexing.
    """

    permutation = permutation_1based - 1
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(
        len(permutation),
        dtype=torch.long,
    )
    return inverse


OPTIMAL_PERMUTATION = invert_qaplib_permutation(
    QAPLIB_OPTIMAL_PERMUTATION
)


# ============================================================
# Nug20 distance matrix
# ============================================================

DISTANCE = torch.tensor(
    [
        [0,1,2,3,4,1,2,3,4,5,2,3,4,5,6,3,4,5,6,7],
        [1,0,1,2,3,2,1,2,3,4,3,2,3,4,5,4,3,4,5,6],
        [2,1,0,1,2,3,2,1,2,3,4,3,2,3,4,5,4,3,4,5],
        [3,2,1,0,1,4,3,2,1,2,5,4,3,2,3,6,5,4,3,4],
        [4,3,2,1,0,5,4,3,2,1,6,5,4,3,2,7,6,5,4,3],
        [1,2,3,4,5,0,1,2,3,4,1,2,3,4,5,2,3,4,5,6],
        [2,1,2,3,4,1,0,1,2,3,2,1,2,3,4,3,2,3,4,5],
        [3,2,1,2,3,2,1,0,1,2,3,2,1,2,3,4,3,2,3,4],
        [4,3,2,1,2,3,2,1,0,1,4,3,2,1,2,5,4,3,2,3],
        [5,4,3,2,1,4,3,2,1,0,5,4,3,2,1,6,5,4,3,2],
        [2,3,4,5,6,1,2,3,4,5,0,1,2,3,4,1,2,3,4,5],
        [3,2,3,4,5,2,1,2,3,4,1,0,1,2,3,2,1,2,3,4],
        [4,3,2,3,4,3,2,1,2,3,2,1,0,1,2,3,2,1,2,3],
        [5,4,3,2,3,4,3,2,1,2,3,2,1,0,1,4,3,2,1,2],
        [6,5,4,3,2,5,4,3,2,1,4,3,2,1,0,5,4,3,2,1],
        [3,4,5,6,7,2,3,4,5,6,1,2,3,4,5,0,1,2,3,4],
        [4,3,4,5,6,3,2,3,4,5,2,1,2,3,4,1,0,1,2,3],
        [5,4,3,4,5,4,3,2,3,4,3,2,1,2,3,2,1,0,1,2],
        [6,5,4,3,4,5,4,3,2,3,4,3,2,1,2,3,2,1,0,1],
        [7,6,5,4,3,6,5,4,3,2,5,4,3,2,1,4,3,2,1,0],
    ],
    dtype=torch.float32,
)


# ============================================================
# Nug20 flow matrix
# ============================================================

FLOW = torch.tensor(
    [
        [0,0,5,0,5,2,10,3,1,5,5,5,0,0,5,4,4,0,0,1],
        [0,0,3,10,5,1,5,1,2,4,2,5,0,10,10,3,0,5,10,5],
        [5,3,0,2,0,5,2,4,4,5,0,0,0,5,1,0,0,5,0,0],
        [0,10,2,0,1,0,5,2,1,0,10,2,2,0,2,1,5,2,5,5],
        [5,5,0,1,0,5,6,5,2,5,2,0,5,1,1,1,5,2,5,1],
        [2,1,5,0,5,0,5,2,1,6,0,0,10,0,2,0,1,0,1,5],
        [10,5,2,5,6,5,0,0,0,0,5,10,2,2,5,1,2,1,0,10],
        [3,1,4,2,5,2,0,0,1,1,10,10,2,0,10,2,5,2,2,10],
        [1,2,4,1,2,1,0,1,0,2,0,3,5,5,0,5,0,0,0,2],
        [5,4,5,0,5,6,0,1,2,0,5,5,0,5,1,0,0,5,5,2],
        [5,2,0,10,2,0,5,10,0,5,0,5,2,5,1,10,0,2,2,5],
        [5,5,0,2,0,0,10,10,3,5,5,0,2,10,5,0,1,1,2,5],
        [0,0,0,2,5,10,2,2,5,0,2,2,0,2,2,1,0,0,0,5],
        [0,10,5,0,1,0,2,0,5,5,5,10,2,0,5,5,1,5,5,0],
        [5,10,1,2,1,2,5,10,0,1,1,5,2,5,0,3,0,5,10,10],
        [4,3,0,1,1,0,1,2,5,0,10,0,1,5,3,0,0,0,2,0],
        [4,0,0,5,5,1,2,5,0,0,0,1,0,1,0,0,0,5,2,0],
        [0,5,5,2,2,0,1,2,0,5,2,1,0,5,5,0,5,0,1,1],
        [0,10,0,5,5,1,0,2,0,5,2,2,0,5,10,2,2,1,0,6],
        [1,5,0,5,1,5,10,10,2,2,5,5,5,0,10,0,0,1,6,0],
    ],
    dtype=torch.float32,
)


def get_nug20(
    device: torch.device | str = "cpu",
):
    return FLOW.to(device), DISTANCE.to(device)


def qap_cost(
    permutations: torch.Tensor,
    flow: torch.Tensor,
    distance: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate one or more QAP permutations.

    Convention:
        permutation[i] = location assigned to facility i

    Lower cost is better.
    """

    if permutations.ndim == 1:
        permutations = permutations.unsqueeze(0)

    permutations = permutations.long()

    assigned_distances = distance[
        permutations.unsqueeze(2),
        permutations.unsqueeze(1),
    ]

    return (
        flow.unsqueeze(0)
        * assigned_distances
    ).sum(dim=(1, 2))


def qap_reward(
    permutations: torch.Tensor,
    flow: torch.Tensor,
    distance: torch.Tensor,
) -> torch.Tensor:
    return -qap_cost(
        permutations,
        flow,
        distance,
    )


def optimality_gap(
    cost: torch.Tensor | float,
) -> torch.Tensor:
    if not torch.is_tensor(cost):
        cost = torch.tensor(
            cost,
            dtype=torch.float32,
        )

    return (
        (cost - OPTIMUM)
        / OPTIMUM
        * 100.0
    )


def qap_cost_slow(
    permutation: torch.Tensor,
    flow: torch.Tensor,
    distance: torch.Tensor,
) -> torch.Tensor:
    n = permutation.numel()

    cost = torch.tensor(
        0.0,
        device=flow.device,
    )

    for i in range(n):
        for j in range(n):
            cost += (
                flow[i, j]
                * distance[
                    permutation[i],
                    permutation[j],
                ]
            )

    return cost


if __name__ == "__main__":

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    flow, distance = get_nug20(
        device=DEVICE,
    )

    optimal_permutation = (
        OPTIMAL_PERMUTATION.to(DEVICE)
    )

    print("=" * 60)
    print("QAPLIB NUG20 SANITY TEST")
    print("=" * 60)

    print()
    print("device:")
    print(DEVICE)

    print()
    print("dimension:")
    print(N)

    print()
    print("flow shape:")
    print(flow.shape)

    print()
    print("distance shape:")
    print(distance.shape)

    print()
    print("QAPLIB optimal permutation")
    print("(location -> facility, 1-based):")
    print(QAPLIB_OPTIMAL_PERMUTATION)

    print()
    print("our optimal permutation")
    print("(facility -> location, 0-based):")
    print(optimal_permutation)

    identity = torch.arange(
        N,
        device=DEVICE,
    )

    identity_cost = qap_cost(
        identity,
        flow,
        distance,
    )[0]

    print()
    print("identity cost:")
    print(identity_cost.item())

    optimum_cost = qap_cost(
        optimal_permutation,
        flow,
        distance,
    )[0]

    print()
    print("computed optimal cost:")
    print(optimum_cost.item())

    print()
    print("known optimum:")
    print(OPTIMUM)

    gap = optimality_gap(
        optimum_cost
    )

    print()
    print("optimality gap:")
    print(gap.item())

    slow_cost = qap_cost_slow(
        optimal_permutation,
        flow,
        distance,
    )

    print()
    print("slow cost:")
    print(slow_cost.item())

    generator = torch.Generator(
        device=DEVICE,
    )
    generator.manual_seed(42)

    random_permutation = torch.randperm(
        N,
        generator=generator,
        device=DEVICE,
    )

    random_cost = qap_cost(
        random_permutation,
        flow,
        distance,
    )[0]

    print()
    print("random permutation:")
    print(random_permutation)

    print()
    print("random cost:")
    print(random_cost.item())

    print()
    print("random gap:")
    print(
        optimality_gap(
            random_cost
        ).item()
    )

    assert flow.shape == (N, N)
    assert distance.shape == (N, N)

    assert torch.allclose(
        flow,
        flow.T,
    )

    assert torch.allclose(
        distance,
        distance.T,
    )

    assert torch.all(
        torch.diag(flow) == 0
    )

    assert torch.all(
        torch.diag(distance) == 0
    )

    assert optimum_cost.item() == OPTIMUM, (
        f"Expected optimum {OPTIMUM}, "
        f"got {optimum_cost.item()}"
    )

    assert torch.allclose(
        optimum_cost,
        slow_cost,
    )

    assert torch.equal(
        torch.sort(
            optimal_permutation
        ).values.cpu(),
        torch.arange(N),
    )

    print()
    print("=" * 60)
    print("All Nug20 checks passed.")
    print("=" * 60)
