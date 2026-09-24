import torch


def jump_target(dimension: int, device=None):
    """
    Nevergrad-style binary target:
    [0, 1, 0, 1, ..., 0, 1]
    """
    return torch.arange(
        dimension,
        device=device,
        dtype=torch.int32,
    ) % 2


def jump_fitness(x: torch.Tensor, jump_size: int | None = None) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)

    dimension = x.shape[1]

    if jump_size is None:
        jump_size = dimension // 4

    target = jump_target(
        dimension,
        device=x.device,
    )

    correct = (x == target).sum(dim=1)

    fitness = torch.where(
        correct == dimension,
        torch.full_like(correct, dimension),
        torch.where(
            correct <= dimension - jump_size,
            correct,
            (dimension - jump_size) - correct,
        ),
    )

    return fitness.to(torch.float32)

if __name__ == "__main__":
    n = 100
    target = jump_target(n)

    def solution_with_correct_bits(k):
        x = 1 - target.clone()

        if k > 0:
            x[:k] = target[:k]

        return x

    for k in [0, 50, 74, 75, 76, 90, 99, 100]:
        x = solution_with_correct_bits(k)
        score = jump_fitness(x).item()

        print(
            f"correct={k:3d} | "
            f"fitness={score:6.1f}"
        )