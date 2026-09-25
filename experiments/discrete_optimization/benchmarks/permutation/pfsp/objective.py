import torch


# ============================================================
# Taillard PFSP benchmark: Ta001
# ============================================================

N_JOBS = 20
N_MACHINES = 5
BEST_KNOWN = 1278
NEH_REFERENCE = 1286

PROCESSING_TIMES = torch.tensor(
    [
        [54, 79, 16, 66, 58],
        [83, 3, 89, 58, 56],
        [15, 11, 49, 31, 20],
        [71, 99, 15, 68, 85],
        [77, 56, 89, 78, 53],
        [36, 70, 45, 91, 35],
        [53, 99, 60, 13, 53],
        [38, 60, 23, 59, 41],
        [27, 5, 57, 49, 69],
        [87, 56, 64, 85, 13],
        [76, 3, 7, 85, 86],
        [91, 61, 1, 9, 72],
        [14, 73, 63, 39, 8],
        [29, 75, 41, 41, 49],
        [12, 47, 63, 56, 47],
        [77, 14, 47, 40, 87],
        [32, 21, 26, 54, 58],
        [87, 86, 75, 77, 18],
        [68, 5, 77, 51, 68],
        [94, 77, 40, 31, 28],
    ],
    dtype=torch.float32,
)


def get_ta001(
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the Ta001 processing-time matrix on the requested device."""
    return PROCESSING_TIMES.to(device)


def pfsp_makespan(
    permutations: torch.Tensor,
    processing_times: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate one or more PFSP schedules.

    `permutations` may contain a full permutation or a partial sequence.
    Partial sequences are useful for constructive heuristics such as NEH.

    Lower is better.
    """
    if permutations.ndim == 1:
        permutations = permutations.unsqueeze(0)

    permutations = permutations.long()

    batch_size, sequence_length = permutations.shape
    n_jobs, n_machines = processing_times.shape

    if sequence_length > n_jobs:
        raise ValueError(
            "Schedule length cannot exceed the number of jobs: "
            f"{sequence_length} > {n_jobs}"
        )

    completion = torch.zeros(
        batch_size,
        n_machines,
        dtype=processing_times.dtype,
        device=processing_times.device,
    )

    for position in range(sequence_length):
        job_times = processing_times[
            permutations[:, position]
        ]

        completion[:, 0] = (
            completion[:, 0]
            + job_times[:, 0]
        )

        for machine in range(1, n_machines):
            completion[:, machine] = (
                torch.maximum(
                    completion[:, machine],
                    completion[:, machine - 1],
                )
                + job_times[:, machine]
            )

    return completion[:, -1]


def pfsp_reward(
    permutations: torch.Tensor,
    processing_times: torch.Tensor,
) -> torch.Tensor:
    return -pfsp_makespan(
        permutations,
        processing_times,
    )


def optimality_gap(
    makespan: torch.Tensor | float,
) -> torch.Tensor:
    if not torch.is_tensor(makespan):
        makespan = torch.tensor(
            makespan,
            dtype=torch.float32,
        )

    return (
        (makespan - BEST_KNOWN)
        / BEST_KNOWN
        * 100.0
    )


def pfsp_makespan_slow(
    permutation: torch.Tensor,
    processing_times: torch.Tensor,
) -> torch.Tensor:
    permutation = permutation.long()
    sequence_length = permutation.numel()
    n_machines = processing_times.shape[1]

    completion = torch.zeros(
        sequence_length,
        n_machines,
        dtype=processing_times.dtype,
        device=processing_times.device,
    )

    for i in range(sequence_length):
        job = permutation[i]

        for machine in range(n_machines):
            previous_job = (
                completion[i - 1, machine]
                if i > 0
                else torch.tensor(
                    0.0,
                    device=processing_times.device,
                )
            )

            previous_machine = (
                completion[i, machine - 1]
                if machine > 0
                else torch.tensor(
                    0.0,
                    device=processing_times.device,
                )
            )

            completion[i, machine] = (
                torch.maximum(
                    previous_job,
                    previous_machine,
                )
                + processing_times[job, machine]
            )

    return completion[-1, -1]


def neh_permutation(
    processing_times: torch.Tensor,
) -> torch.Tensor:
    """Standard deterministic NEH constructive heuristic."""
    device = processing_times.device
    n_jobs = processing_times.shape[0]

    totals = processing_times.sum(dim=1)

    job_order = sorted(
        range(n_jobs),
        key=lambda job: (
            -float(totals[job].item()),
            job,
        ),
    )

    sequence: list[int] = []

    for job in job_order:
        best_sequence = None
        best_makespan = None

        for position in range(
            len(sequence) + 1
        ):
            candidate = (
                sequence[:position]
                + [job]
                + sequence[position:]
            )

            candidate_tensor = torch.tensor(
                candidate,
                dtype=torch.long,
                device=device,
            )

            candidate_makespan = float(
                pfsp_makespan(
                    candidate_tensor,
                    processing_times,
                )[0].item()
            )

            if (
                best_makespan is None
                or candidate_makespan
                < best_makespan
            ):
                best_makespan = candidate_makespan
                best_sequence = candidate

        sequence = best_sequence

    return torch.tensor(
        sequence,
        dtype=torch.long,
        device=device,
    )


def is_valid_permutation(
    permutation: torch.Tensor,
    n_jobs: int = N_JOBS,
) -> bool:
    if permutation.ndim != 1:
        return False

    if permutation.numel() != n_jobs:
        return False

    expected = torch.arange(
        n_jobs,
        device=permutation.device,
    )

    return bool(
        torch.equal(
            torch.sort(permutation.long()).values,
            expected,
        )
    )


if __name__ == "__main__":

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    processing_times = get_ta001(
        device=DEVICE,
    )

    print("=" * 60)
    print("TAILLARD TA001 PFSP SANITY TEST")
    print("=" * 60)

    print()
    print("device:")
    print(DEVICE)

    print()
    print("jobs:")
    print(N_JOBS)

    print()
    print("machines:")
    print(N_MACHINES)

    print()
    print("processing-times shape:")
    print(processing_times.shape)

    print()
    print("best-known makespan:")
    print(BEST_KNOWN)

    identity = torch.arange(
        N_JOBS,
        device=DEVICE,
    )

    identity_makespan = pfsp_makespan(
        identity,
        processing_times,
    )[0]

    print()
    print("identity permutation:")
    print(identity)

    print()
    print("identity makespan:")
    print(identity_makespan.item())

    identity_slow = pfsp_makespan_slow(
        identity,
        processing_times,
    )

    print()
    print("identity slow makespan:")
    print(identity_slow.item())

    neh = neh_permutation(
        processing_times,
    )

    neh_makespan = pfsp_makespan(
        neh,
        processing_times,
    )[0]

    print()
    print("NEH permutation:")
    print(neh)

    print()
    print("NEH makespan:")
    print(neh_makespan.item())

    print()
    print("NEH gap (%):")
    print(
        optimality_gap(
            neh_makespan
        ).item()
    )

    generator = torch.Generator(
        device=DEVICE,
    )
    generator.manual_seed(42)

    random_permutation = torch.randperm(
        N_JOBS,
        generator=generator,
        device=DEVICE,
    )

    random_fast = pfsp_makespan(
        random_permutation,
        processing_times,
    )[0]

    random_slow = pfsp_makespan_slow(
        random_permutation,
        processing_times,
    )

    print()
    print("random permutation:")
    print(random_permutation)

    print()
    print("random makespan:")
    print(random_fast.item())

    print()
    print("random gap (%):")
    print(
        optimality_gap(
            random_fast
        ).item()
    )

    assert processing_times.shape == (
        N_JOBS,
        N_MACHINES,
    )

    assert torch.all(
        processing_times > 0
    )

    assert is_valid_permutation(
        identity
    )

    assert is_valid_permutation(
        neh
    )

    assert identity_makespan.item() == 1448.0, (
        "Ta001 data/evaluator mismatch: "
        f"identity makespan is {identity_makespan.item()}, "
        "expected 1448."
    )

    assert neh_makespan.item() == float(
        NEH_REFERENCE
    ), (
        "NEH reference mismatch: "
        f"got {neh_makespan.item()}, "
        f"expected {NEH_REFERENCE}."
    )

    assert torch.allclose(
        identity_makespan,
        identity_slow,
    )

    assert torch.allclose(
        random_fast,
        random_slow,
    )

    print()
    print("=" * 60)
    print("All Ta001 checks passed.")
    print("=" * 60)
