import argparse
import copy
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

#spremeljikve

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

N_CITIES = 50

NUM_LAYERS = 8
HIDDEN_DIM = 64

BATCH_SIZE = 4096
EPOCHS = 3000
LR = 1e-4

VALIDATION_SIZE = 4096
VALIDATE_EVERY = 50

TEST_SIZE = 16384

MAX_GRAD_NORM = 5.0

SEED = 42
TSP_INSTANCE_SEED = 12345

ELITE_FRACTION = 0.01


#tsp probelm

def create_tsp_instance(n_cities, seed):
  
    rng = np.random.default_rng(seed)

    cities = rng.random(
        (n_cities, 2)
    )

    return torch.tensor(
        cities,
        dtype=torch.float32,
        device=DEVICE,
    )


def decode_permutation(y):
    return torch.argsort(y, dim=1)

@torch.no_grad()
def tsp_length(cities, tours):
    """
    Compute closed TSP tour length.

    cities:
        [N_CITIES, 2]

    tours:
        [BATCH, N_CITIES]

    returns:
        [BATCH]
    """

    ordered_cities = cities[
        tours
    ]

    next_cities = torch.roll(
        ordered_cities,
        shifts=-1,
        dims=1,
    )

    edge_lengths = torch.linalg.vector_norm(
        ordered_cities - next_cities,
        dim=2,
    )

    return edge_lengths.sum(
        dim=1
    )


# ============================================================
# COSINE LEARNING RATE
# ============================================================

def learning_rate_at(epoch, epochs):

    progress = (
        epoch
        / max(
            epochs - 1,
            1
        )
    )

    min_lr = LR / 10
    return (
        min_lr
        + (
            LR - min_lr
        )
        * (
            1
            + math.cos(
                math.pi * progress
            )
        )
        / 2
    )


#real nvp

class CouplingLayer(nn.Module):

    def __init__(
        self,
        dimension,
        hidden_dim,
    ):
        super().__init__()

        if dimension % 2 != 0:
            raise ValueError(
                "This RealNVP implementation requires "
                "an even dimension."
            )

        self.half = (
            dimension // 2
        )

        self.scale_net = nn.Sequential(
            nn.Linear(
                self.half,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                self.half,
            ),
            nn.Tanh(),
        )

        self.translate_net = nn.Sequential(
            nn.Linear(self.half, hidden_dim,),
            nn.ReLU(),
            nn.Linear( hidden_dim,hidden_dim, ),
            nn.ReLU(),
            nn.Linear(hidden_dim,self.half, ),
        )

    def forward(self, x):

        x1 = x[:, :self.half]
        x2 = x[:,  self.half:]

        s = self.scale_net(x1)

        t = self.translate_net( x1 )

        y1 = x1
        y2 = ( x2 * torch.exp(s) + t )

        y = torch.cat(
            [ y1, y2, ],
            dim=1,
        )

        log_det = s.sum( dim=1 )

        return (
            y,
            log_det,
        )

    def inverse(self, y):

        y1 = y[:,:self.half]

        y2 = y[:,self.half: ]

        s = self.scale_net(y1)

        t = self.translate_net(y1)

        x1 = y1

        x2 = (y2 - t) * torch.exp(-s)

        x = torch.cat(
            [x1,x2,],
            dim=1,
        )

        inverse_log_det = (
            -s.sum(
                dim=1
            )
        )

        return (
            x,
            inverse_log_det,
        )


class RealNVP(nn.Module):

    def __init__(
        self,
        dimension,
        num_layers,
        hidden_dim,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                CouplingLayer(
                    dimension,
                    hidden_dim,
                )
                for _ in range(
                    num_layers
                )
            ]
        )

        self.half = (
            dimension // 2
        )

    def _swap(self, x):

        return torch.cat(
             [x[:,self.half:],
               x[:,:self.half],
            ],
            dim=1,
        )
         
    def forward(self, x):

        y = x

        log_det_total = x.new_zeros(
            x.shape[0]
        )

        for layer in self.layers:

            y, log_det = layer(y)

            log_det_total += (log_det)

            y = self._swap(y)

        return (
            y,
            log_det_total,
        )

    def inverse(self, y):
        x = y

        log_det_total = y.new_zeros(
            y.shape[0]
        )

        for layer in reversed(
            self.layers
        ):

            x = self._swap(x)

            x, log_det = layer.inverse(x)

            log_det_total += (log_det)

        return (
            x,
            log_det_total,
        )


def log_probability(model, y):
    

    z, inverse_log_det = (
        model.inverse(y)
    )

    log_pz = -0.5 * (
        z.square()
        + math.log(
            2.0 * math.pi
        )
    ).sum(
        dim=1
    )

    return (
        log_pz
        + inverse_log_det
    )


#plot

def plot_tour_comparison(
    cities,
    best_tour,
    best_length,
    test_best_tour,
    test_best_length,
    filename,
):

    if isinstance(
        cities,
        torch.Tensor
    ):
        cities = (
            cities
            .detach()
            .cpu()
            .numpy()
        )

    if isinstance(
        best_tour,
        torch.Tensor
    ):
        best_tour = (
            best_tour
            .detach()
            .cpu()
            .numpy()
        )

    if isinstance(
        test_best_tour,
        torch.Tensor
    ):
        test_best_tour = (
            test_best_tour
            .detach()
            .cpu()
            .numpy()
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    tours = [
        (
            best_tour,
            best_length,
            "Best ever",
        ),
        (
            test_best_tour,
            test_best_length,
            "Final test best",
        ),
    ]

    for ax, (
        tour,
        length,
        title,
    ) in zip(
        axes,
        tours,
    ):
        ordered = cities[
            tour
        ]

        closed = np.vstack(
            [
                ordered,
                ordered[0],
            ]
        )

        ax.plot(
            closed[:, 0],
            closed[:, 1],
            marker="o",
        )

        for i, (
            x,
            y,
        ) in enumerate(
            cities
        ):

            ax.text(
                x,
                y,
                str(i),
                fontsize=9,
            )

        ax.set_title(
            f"{title}\n"
            f"Length = {length:.6f}"
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.set_aspect(
            "equal"
        )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=200,
    )

    plt.close()

def plot_convergence(
    history_mean,
    history_best,
    output_dir,
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history_mean,
        label="Mean sampled tour length",
    )

    plt.plot(
        history_best,
        label="Best-so-far tour length",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Tour length")

    plt.title(
        "RealNVP TSP-20\n"
        "Cosine LR + Elite Validation Checkpoint"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir
        / "realnvp_tsp_convergence.png",
        dpi=200,
    )

    plt.close()

#training

def train(
    seed=SEED,
    epochs=EPOCHS,
    output_dir=None,
    plots=True,
):

    if epochs < 1:
        raise ValueError(
            "epochs must be positive"
        )

    output_dir = Path(
        output_dir
        or (
            "results/"
            "tsp_v2a_cosine_elite/"
            f"seed_{seed}"
        )
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    torch.manual_seed(seed)

    np.random.seed(seed )

  

    cities = create_tsp_instance(
        N_CITIES,
        TSP_INSTANCE_SEED,
    )


    model = RealNVP(
        dimension=N_CITIES,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
    ).to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )


    validation_generator = (
        torch.Generator(
            device=DEVICE
        )
        .manual_seed(
            seed + 1000
        )
    )

    validation = torch.randn(
        VALIDATION_SIZE,
        N_CITIES,
        generator=validation_generator,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Checkpoint state
    # --------------------------------------------------------

    # Mean is retained only for diagnostics.
    best_validation_mean = (
        float("inf")
    )

    # actual checkpoint criterion.
    best_validation_elite_mean = (
        float("inf")
    )
    best_checkpoint_validation_mean = float("inf")
    best_state = None
    best_epoch = -1


    best_length = float(
        "inf"
    )

    best_tour = None

    

    history_mean = []
    history_best = []

    diagnostics = []

    evaluations = 0

    current_val_mean = float(
        "nan"
    )

    current_val_elite_mean = float(
        "nan"
    )

    current_val_best = float(
        "nan"
    )


    for epoch in range(
        epochs
    ):

       #LR
        learning_rate = learning_rate_at(
            epoch,
            epochs,
        )

        for group in (
            optimizer.param_groups
        ):
            group["lr"] = (
                learning_rate
            )


        r = torch.randn(
            BATCH_SIZE,
            N_CITIES,
            device=DEVICE,
        )

       

        with torch.no_grad():

            y, _ = model(r)

            tours = decode_permutation( y)

            lengths = tsp_length(
                cities,
                tours,
            )

            # TSP minimizes length.
            # REINFORCE maximizes reward.
            reward = -lengths

        evaluations += (
            BATCH_SIZE
        )


        batch_best_index = (
            torch.argmin(
                lengths
            )
        )

        batch_best_length = (
            lengths[
                batch_best_index
            ].item()
        )

        if (
            batch_best_length
            < best_length
        ):

            best_length = (
                batch_best_length
            )

            best_tour = (
                tours[
                    batch_best_index
                ]
                .detach()
                .cpu()
                .clone()
            )


        with torch.no_grad():

            baseline = ( reward.sum() - reward) / (BATCH_SIZE - 1)

            advantage = (reward- baseline)


        log_prob = log_probability( model,y.detach(),)

        loss = -(advantage* log_prob).mean()

        optimizer.zero_grad()

        loss.backward()

        grad_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )
        )

        optimizer.step()

        

        mean_length = (
            lengths.mean().item()
        )

        history_mean.append(
            mean_length
        )

        #validation
        if (
            epoch % VALIDATE_EVERY == 0
            or epoch == epochs - 1
        ):

            with torch.no_grad():

                val_y, _ = model(
                    validation
                )

                val_tours = (
                    decode_permutation(
                        val_y
                    )
                )

                val_lengths = (
                    tsp_length(
                        cities,
                        val_tours,
                    )
                )
                evaluations += (
                    VALIDATION_SIZE
                )

                val_mean = (
                    val_lengths
                    .mean()
                    .item()
                )

                val_best_index = (
                    torch.argmin(
                        val_lengths
                    )
                )

                val_best_length = (
                    val_lengths[
                        val_best_index
                    ].item()
                )

                if (
                    val_best_length
                    < best_length
                ):

                    best_length = (
                        val_best_length
                    )

                    best_tour = (
                        val_tours[
                            val_best_index
                        ]
                        .detach()
                        .cpu()
                        .clone()
                    )

                # NEW V2A:
                # TOP 1 % VALIDATION PERFORMANCE
        

                sorted_val_lengths = (
                    torch.sort(
                        val_lengths
                    ).values
                )

                elite_k = max(
                    1,
                    int(
                        ELITE_FRACTION
                        * VALIDATION_SIZE
                    ),
                )

                val_elite_mean = (
                    sorted_val_lengths[
                        :elite_k
                    ]
                    .mean()
                    .item()
                )

                
                #diagnostic

                current_val_mean = (
                    val_mean
                )

                current_val_elite_mean = (
                    val_elite_mean
                )

                current_val_best = (
                    val_best_length
                )

            if (
                val_mean
                < best_validation_mean
            ):

                best_validation_mean = (
                    val_mean
                )
       
            # ACTUAL CHECKPOINT CRITERION
            #
            # Best 1 % validation mean.
            # If elite performance is tied, choose the
            # checkpoint with the better overall val mean.
   
            ELITE_EPS = 1e-6

            elite_improved = (
                val_elite_mean
                < best_validation_elite_mean - ELITE_EPS
            )

            elite_tied = (
                abs(
                    val_elite_mean
                    - best_validation_elite_mean
                )
                <= ELITE_EPS
            )

            mean_improved_on_tie = (
                elite_tied
                and val_mean
                < best_checkpoint_validation_mean
            )

            if (
                elite_improved
                or mean_improved_on_tie
            ):

                best_validation_elite_mean = (
                    val_elite_mean
                )

                best_checkpoint_validation_mean = (
                    val_mean
                )

                best_epoch = epoch

                best_state = copy.deepcopy(
                    model.state_dict()
                )

        history_best.append(
            best_length
        )

      

        if (
            epoch % 50 == 0
            or epoch == epochs - 1
        ):

            row = dict(
                epoch=epoch,
                updates=epoch + 1,
                evaluations=evaluations,

                learning_rate=learning_rate,

                mean_length=mean_length,

                length_std=(
                    lengths
                    .std(
                        unbiased=False
                    )
                    .item()
                ),

                validation_mean=(
                    current_val_mean
                ),

                validation_elite_mean=(
                    current_val_elite_mean
                ),

                validation_best=(
                    current_val_best
                ),

                grad_norm_before_clip=(
                    grad_norm.item()
                ),

                best_length=(
                    best_length
                ),

                loss=(
                    loss.item()
                ),
            )

            diagnostics.append(
                row
            )

            print(
                f"{epoch:5d} | "
                f"mean={mean_length:.6f} | "
                f"best={best_length:.6f} | "
                f"val_mean={current_val_mean:.6f} | "
                f"val_elite={current_val_elite_mean:.6f} | "
                f"val_best={current_val_best:.6f} | "
                f"std={row['length_std']:.4f} | "
                f"grad={grad_norm.item():.4f} | "
                f"lr={learning_rate:.2e}"
            )
    # RESTORE BEST ELITE CHECKPOINT

    optimization_best_length = (
        best_length
    )
    if best_state is not None:

        model.load_state_dict(
            best_state
        )
    # FINAL TEST
    test_generator = (
        torch.Generator(
            device=DEVICE
        )
        .manual_seed(
            seed + 2000
        )
    )

    test = torch.randn(
        TEST_SIZE,
        N_CITIES,
        generator=test_generator,
        device=DEVICE,
    )

    with torch.no_grad():

        test_y, _ = model(
            test
        )

        test_tours = (
            decode_permutation(
                test_y
            )
        )

        test_lengths = (
            tsp_length(
                cities,
                test_tours,
            )
        )

    test_best_index = (
        torch.argmin(
            test_lengths
        )
    )

    test_best_length = (
        test_lengths[
            test_best_index
        ].item()
    )

    test_best_tour = (
        test_tours[
            test_best_index
        ]
        .detach()
        .cpu()
        .clone()
    )

    test_mean_length = (
        test_lengths
        .mean()
        .item()
    )


    if plots:

        plot_tour_comparison(
            cities,
            best_tour,
            best_length,
            test_best_tour,
            test_best_length,
            filename=(
                output_dir
                / "tsp_tour_comparison.png"
            ),
        )

        plot_convergence(
            history_mean,
            history_best,
            output_dir,
        )
    # Include final test in absolute best reported solution.
    if (
        test_best_length
        < best_length
    ):

        best_length = (
            test_best_length
        )

        best_tour = (
            test_best_tour.clone()
        )
    # FINAL OUTPUT
    print()

    print(
        "=" * 75
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 75
    )

    print(
        f"Best elite checkpoint epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation mean seen:   "
        f"{best_validation_mean:.6f}"
    )

    print(
        f"Best validation elite mean:  "
        f"{best_validation_elite_mean:.6f}"
    )

    print(
        f"Elite fraction:               "
        f"{ELITE_FRACTION:.2%}"
    )

    print(
        f"Test mean length:             "
        f"{test_mean_length:.6f}"
    )

    print(
        f"Test best length:             "
        f"{test_best_length:.6f}"
    )

    print(
        f"Optimization best length:     "
        f"{optimization_best_length:.6f}"
    )

    print(
        f"Best length including test:   "
        f"{best_length:.6f}"
    )

    print(
        f"Best tour:                    "
        f"{best_tour.tolist()}"
    )

    diagnostics_path = (
        output_dir
        / "diagnostics.csv"
    )

    with diagnostics_path.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=(
                diagnostics[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            diagnostics
        )

    summary = dict(

        experiment=(
            "tsp_v2a_cosine_elite_checkpoint"
        ),

        seed=seed,

        epochs=epochs,

        initial_lr=LR,

        final_lr=learning_rate,

        lr_schedule="cosine",

        checkpoint_metric=(
            "validation_top_1pct_mean"
        ),

        elite_fraction=(
            ELITE_FRACTION
        ),

        optimization_evaluations=(
            evaluations
        ),

        test_evaluations=(
            TEST_SIZE
        ),

        optimization_best_length=(
            optimization_best_length
        ),

        best_length_including_test=(
            best_length
        ),

        test_best_length=(
            test_best_length
        ),

        test_mean_length=(
            test_mean_length
        ),

        best_validation_mean=(
            best_validation_mean
        ),

        best_validation_elite_mean=(
            best_validation_elite_mean
        ),

        best_checkpoint_epoch=(
            best_epoch
        ),

        n_cities=N_CITIES,

        instance_seed=(
            TSP_INSTANCE_SEED
        ),

        batch_size=(
            BATCH_SIZE
        ),

        validation_size=(
            VALIDATION_SIZE
        ),

        validate_every=(
            VALIDATE_EVERY
        ),

        num_layers=(
            NUM_LAYERS
        ),

        hidden_dim=(
            HIDDEN_DIM
        ),

        max_grad_norm=(
            MAX_GRAD_NORM
        ),

        device=str(
            DEVICE
        ),

        torch_version=(
            torch.__version__
        ),
        best_checkpoint_validation_mean=(
    best_checkpoint_validation_mean
),
    )   

    summary_path = (
        output_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    return (
        model,
        cities.detach().cpu(),
        best_tour,
        history_mean,
        history_best,
        summary,
    )


# MAIN


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "RealNVP TSP-20 : "
            "cosine learning rate + "
            "top-1% validation checkpoint."
        )
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[SEED],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/"
            "tsp_v2a_cosine_elite"
        ),
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error(
            "epochs must be positive"
        )

    if args.threads < 1:
        parser.error(
            "threads must be positive"
        )

    torch.set_num_threads(
        args.threads
    )

    summaries = []

    for seed in args.seeds:

        destination = (
            args.output_dir
            / f"seed_{seed}"
        )

        print()
        print(
            "=" * 75
        )

        print(
            f"TSP v2a | "
            f"seed={seed} | "
            f"cosine LR | "
            f"elite checkpoint={ELITE_FRACTION:.1%}"
        )

        print(
            "=" * 75
        )

        (
            model,
            cities,
            best_tour,
            history_mean,
            history_best,
            summary,
        ) = train(
            seed=seed,
            epochs=args.epochs,
            output_dir=destination,
            plots=not args.no_plots,
        )

        summaries.append(
            summary
        )


    # MULTI-SEED COMPARISON CSV
    

    comparison_path = (
        args.output_dir
        / "comparison.csv"
    )

    comparison_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with comparison_path.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=(
                summaries[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            summaries
        )

    print()
    print(
        f"Saved comparison: "
        f"{comparison_path}"
    )
    print(
    f"Checkpoint validation mean:   "
    f"{summary['best_checkpoint_validation_mean']:.6f}"
)
