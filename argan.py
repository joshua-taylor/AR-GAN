##updated with grid search and comparing Gan against itself...

#!/usr/bin/env python3
# argan_grid_benchmark.py
"""
Grid-style benchmark: runs AR-GAN against itself with different warmup schedules,
data sizes, and architectures (sequential).
"""

import os
import csv
import time
import random
from itertools import product
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------------------------
# Repro & device
# ---------------------------
def set_global_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Data generator (raw)
# ---------------------------
def generate_benchmark_data(
    task_type: str = "classification",
    n_samples: int = 1000,
    n_features: int = 20,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return raw numpy X, y (unscaled)."""
    rng = np.random.RandomState(seed)
    if task_type == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.5), # Increase informative features
            n_redundant=0, # Eliminate redundant features
            n_clusters_per_class=1, # Use a single cluster per class
            flip_y=0.01, # Reduce label noise
            class_sep=1.0, # Increase separation between classes
            random_state=seed,
        )
    else:
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=seed)

    # make slightly harder with additional Gaussian noise (deterministic via rng)
    X = X + rng.normal(0, 0.7, X.shape)
    return X.astype(np.float32), y.astype(np.int64 if task_type == "classification" else np.float32)

# ---------------------------
# Model components (TaskNetwork, Critic, ARGAN)
# ---------------------------
class TaskNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        d = input_dim
        for h in hidden_dims:
            self.layers.append(nn.Linear(d, h))
            self.activations.append(nn.ReLU())
            self.dropouts.append(nn.Dropout(dropout_rate))
            d = h
        self.output_layer = nn.Linear(d, output_dim)
        self.hidden_dims = hidden_dims

    def forward(self, x: torch.Tensor, keep_probs: Optional[List[torch.Tensor]] = None):
        activations_history: List[torch.Tensor] = []
        for i, (lin, act, drp) in enumerate(zip(self.layers, self.activations, self.dropouts)):
            x = lin(x)
            x = act(x)
            activations_history.append(x)
            if keep_probs is not None and i < len(keep_probs) and keep_probs[i] is not None:
                p = keep_probs[i]
                keep = p.mean(dim=1, keepdim=True).clamp_min(1e-6)
                x = x * p / keep
            else:
                x = drp(x)
        logits = self.output_layer(x)
        return logits, activations_history

class RegularizationCritic(nn.Module):
    def __init__(self, hidden_dims: List[int], context_dim: int = 10):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.context_dim = context_dim
        self.mask_generators = nn.ModuleList()
        for h in hidden_dims:
            self.mask_generators.append(
                nn.Sequential(
                    nn.Linear(h + context_dim, h * 2),
                    nn.ReLU(),
                    nn.Linear(h * 2, h),
                    nn.Sigmoid(),  # produce keep-probabilities
                )
            )
        total_hidden = sum(hidden_dims)
        self.critic = nn.Sequential(
            nn.Linear(total_hidden + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # logits (BCEWithLogitsLoss)
        )

    def generate_keep_probs(self, activations: List[torch.Tensor], context: torch.Tensor) -> List[torch.Tensor]:
        probs: List[torch.Tensor] = []
        B = activations[0].shape[0]
        for i, a in enumerate(activations):
            act_mean = a.mean(dim=0, keepdim=True).expand(B, -1)
            inp = torch.cat([act_mean, context], dim=1)
            p = self.mask_generators[i](inp).clamp(1e-4, 1 - 1e-4)
            probs.append(p)
        return probs

    def evaluate_regularization(self, activations: List[torch.Tensor], context: torch.Tensor) -> torch.Tensor:
        B = context.shape[0]
        flat = [a.mean(dim=0) for a in activations]
        combined = torch.cat(flat).unsqueeze(0).expand(B, -1)
        inp = torch.cat([combined, context], dim=1)
        return self.critic(inp)  # logits

class ARGAN:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        task_type: str = "classification",
        dropout_rate: float = 0.5,
        warmup_epochs: int = 30,
        target_keep: float = 0.5,
    ):
        self.task_type = task_type
        self.task_network = TaskNetwork(input_dim, hidden_dims, output_dim, dropout_rate).to(device)
        self.critic = RegularizationCritic(hidden_dims).to(device)

        # separate optimizers
        self.critic_optimizer = optim.Adam(self._critic_params(), lr=1e-4)
        self.task_optimizer = optim.Adam(
            list(self.task_network.parameters()) + list(self.critic.mask_generators.parameters()), lr=1e-3
        )

        self.task_loss_fn = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
        self.adv_loss_fn = nn.BCEWithLogitsLoss()

        self.warmup_epochs = warmup_epochs
        self.adversarial_weight = 0.01
        self.target_keep = target_keep

        self.metrics: Dict[str, List[float]] = {
            "train_task_loss": [],
            "train_task_accuracy": [],
            "val_task_loss": [],
            "val_task_accuracy": [],
            "critic_loss": [],
            "sparsity": [],
        }

    def _critic_params(self):
        return self.critic.critic.parameters()

    def _create_context(self, task_loss: torch.Tensor, activations: List[torch.Tensor], batch_size: int) -> torch.Tensor:
        act_means = [a.mean().item() for a in activations]
        act_stds = [a.std().item() for a in activations]
        ctx = [
            float(task_loss.detach().item()),
            float(np.mean(act_means)),
            float(np.mean(act_stds)),
            float(np.max(act_means)),
            float(np.min(act_means)),
            float(len(activations)),
            float(np.std(act_means)),
            *np.random.normal(0, 0.1, 3).tolist(),
        ]
        context = torch.tensor(ctx, dtype=torch.float32, device=device)
        return context.unsqueeze(0).expand(batch_size, -1)

    def _accuracy(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        if self.task_type == "classification":
            return (logits.argmax(dim=1) == y).float().mean().item()
        else:
            loss = self.task_loss_fn(logits.squeeze(-1), y)
            return float(1.0 / (1.0 + loss.item()))

    @torch.no_grad()
    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        self.task_network.eval()
        logits, _ = self.task_network(X)
        loss = self.task_loss_fn(logits, y).item()
        acc = self._accuracy(logits, y)
        self.task_network.train()
        return loss, acc

    def _train_step(self, x: torch.Tensor, y: torch.Tensor, epoch: int) -> Dict[str, float]:
        B = x.shape[0]

        # Warmup
        if epoch < self.warmup_epochs:
            self.task_optimizer.zero_grad()
            logits, activations = self.task_network(x)
            task_loss = self.task_loss_fn(logits, y)
            task_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.task_network.parameters(), 1.0)
            self.task_optimizer.step()

            keep = 1.0 - self.task_network.dropouts[0].p if len(self.task_network.dropouts) else 1.0
            sparsity = float(1.0 - keep)
            critic_loss_val = 0.0

            return {
                "task_loss": float(task_loss.item()),
                "task_accuracy": self._accuracy(logits, y),
                "critic_loss": critic_loss_val,
                "sparsity": sparsity,
            }

        # Adversarial phase
        self.task_optimizer.zero_grad()
        logits0, activations0 = self.task_network(x)
        task_loss0 = self.task_loss_fn(logits0, y)
        context = self._create_context(task_loss0, activations0, B)

        keep_probs = self.critic.generate_keep_probs(activations0, context)
        alpha = 0.5
        keep_probs = [alpha * p + (1 - alpha) * self.target_keep for p in keep_probs]

        logits, activations = self.task_network(x, keep_probs=keep_probs)
        task_loss = self.task_loss_fn(logits, y)

        crit_logits = self.critic.evaluate_regularization(activations, context)
        adv_loss = self.adv_loss_fn(crit_logits, torch.ones_like(crit_logits))

        mean_keep = torch.stack([p.mean() for p in keep_probs]).mean()
        keep_reg = (mean_keep - self.target_keep).abs()

        total_task_loss = task_loss + self.adversarial_weight * adv_loss + 1e-5  * keep_reg
        total_task_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.task_network.parameters()) + list(self.critic.mask_generators.parameters()), 1.0
        )
        self.task_optimizer.step()

        critic_loss_val = 0.0
        if epoch % 3 == 0:
            self.critic_optimizer.zero_grad()
            with torch.no_grad():
                good_keep, bad_keep = 0.8, 0.3
                good_keep_probs = [torch.full_like(a, good_keep) for a in activations0]
                bad_keep_probs = [torch.full_like(a, bad_keep) for a in activations0]
                _, good_acts = self.task_network(x, keep_probs=good_keep_probs)
                good_loss = self.task_loss_fn(self.task_network.output_layer(good_acts[-1]), y)
                _, bad_acts = self.task_network(x, keep_probs=bad_keep_probs)
                bad_loss = self.task_loss_fn(self.task_network.output_layer(bad_acts[-1]), y)
                ctx_good = self._create_context(good_loss, good_acts, B)
                ctx_bad = self._create_context(bad_loss, bad_acts, B)

            pos_logits = self.critic.evaluate_regularization(good_acts, ctx_good)
            neg_logits = self.critic.evaluate_regularization(bad_acts, ctx_bad)
            critic_loss = 0.5 * (
                self.adv_loss_fn(pos_logits, torch.ones_like(pos_logits))
                + self.adv_loss_fn(neg_logits, torch.zeros_like(neg_logits))
            )
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._critic_params(), 1.0)
            self.critic_optimizer.step()
            critic_loss_val = float(critic_loss.item())

        sparsity = float((1.0 - mean_keep).item())
        return {
            "task_loss": float(task_loss.item()),
            "task_accuracy": self._accuracy(logits, y),
            "critic_loss": critic_loss_val,
            "sparsity": sparsity,
        }

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        shuffle_seed: Optional[int] = None,
    ) -> None:
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        if shuffle_seed is None:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            # use a deterministic generator for DataLoader shuffling
            g = torch.Generator()
            # DataLoader generator runs on CPU RNG
            g.manual_seed(int(shuffle_seed))
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

        for epoch in range(epochs):
            epoch_task_losses, epoch_accs, epoch_spars, epoch_closs = [], [], [], []
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                out = self._train_step(xb, yb, epoch)
                epoch_task_losses.append(out["task_loss"])
                epoch_accs.append(out["task_accuracy"])
                epoch_spars.append(out["sparsity"])
                epoch_closs.append(out["critic_loss"])

            self.metrics["train_task_loss"].append(float(np.mean(epoch_task_losses)))
            self.metrics["train_task_accuracy"].append(float(np.mean(epoch_accs)))
            self.metrics["sparsity"].append(float(np.mean(epoch_spars)))
            self.metrics["critic_loss"].append(float(np.mean(epoch_closs)))

            if X_val is not None and y_val is not None:
                vloss, vacc = self.evaluate(X_val, y_val)
                self.metrics["val_task_loss"].append(vloss)
                self.metrics["val_task_accuracy"].append(vacc)

            if epoch > self.warmup_epochs:
                self.adversarial_weight = min(0.05, self.adversarial_weight * 1.01)

            if epoch % 20 == 0:
                phase = "WARMUP" if epoch < self.warmup_epochs else "ADVERSARIAL"
                tr_loss = self.metrics["train_task_loss"][-1]
                tr_acc = self.metrics["train_task_accuracy"][-1]
                val_str = ""
                if X_val is not None and y_val is not None and len(self.metrics["val_task_accuracy"]) > 0:
                    val_str = f", Val Loss: {self.metrics['val_task_loss'][-1]:.4f}, Val Acc: {self.metrics['val_task_accuracy'][-1]:.4f}"
                print(f"Epoch {epoch:03d} ({phase}) | Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}, Sparsity: {self.metrics['sparsity'][-1]:.4f}{val_str}")

# ---------------------------
# Grid benchmark runner
# ---------------------------
def run_grid_benchmark(
    task_type: str = "classification",
    n_samples_grid: List[int] = [2000, 10000, 50000],
    n_features: int = 20,
    hidden_dims_grid: List[List[int]] = [[6, 4, 2], [64, 32], [128, 64]],
    dropout_rate: float = 0.5,
    epochs: int = 100,
    batch_size: int = 32,
    warmup_grid: Optional[List[int]] = None,
    num_repeats: int = 3,
    out_dir: str = "argan_grid_results",
):
    os.makedirs(out_dir, exist_ok=True)
    if warmup_grid is None:
        warmup_grid = [epochs, max(1, epochs // 2)]

    csv_rows = []
    total_configs = len(n_samples_grid) * len(hidden_dims_grid) * len(warmup_grid) * num_repeats
    config_counter = 0

    # iterate grid
    for n_samples, hidden_dims in product(n_samples_grid, hidden_dims_grid):
        config_name = f"ns{n_samples}_hd{'-'.join(map(str,hidden_dims))}"
        print("\n" + "=" * 60)
        print(f"CONFIG: samples={n_samples}, hidden_dims={hidden_dims}, epochs={epochs}, batch={batch_size}")
        print("=" * 60)

        # repeat experiments to average
        # We'll store per-warmup per-repeat metrics
        results_per_warmup: Dict[int, Dict[str, List[np.ndarray]]] = {}
        for w in warmup_grid:
            results_per_warmup[w] = {
                "val_acc": [],    # list of arrays (repeats x epochs)
                "val_loss": [],
                "sparsity": [],   # list of arrays
                "final_acc": [],  # scalars
                "time_s": [],
            }

        for repeat in range(num_repeats):
            config_counter += 1
            # fix seeds carefully: dataset seed different across repeats
            base_seed = 10000 + hash((n_samples, tuple(hidden_dims), repeat)) % 100000
            set_global_seed(base_seed)

            # generate data once per repeat (so warmup variants use same data)
            X_raw, y_raw = generate_benchmark_data(task_type=task_type, n_samples=n_samples, n_features=n_features, seed=base_seed)
            split = int(0.8 * n_samples)
            X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
            y_train_np, y_test_np = y_raw[:split], y_raw[split:]

            scaler = StandardScaler().fit(X_train_raw)
            X_train_np = scaler.transform(X_train_raw)
            X_test_np = scaler.transform(X_test_raw)

            X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
            X_test = torch.tensor(X_test_np, dtype=torch.float32, device=device)
            if task_type == "classification":
                y_train = torch.tensor(y_train_np, dtype=torch.long, device=device)
                y_test = torch.tensor(y_test_np, dtype=torch.long, device=device)
                output_dim = int(np.unique(y_raw).shape[0])
            else:
                y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)
                y_test = torch.tensor(y_test_np, dtype=torch.float32, device=device)
                output_dim = 1

            # For fairness: use same model init & same DataLoader shuffle seed for all warmup variants in this repeat.
            # derive stable seeds
            model_seed = base_seed + 12345
            shuffle_seed = base_seed + 54321

            for w in warmup_grid:
                config_counter += 1
                print(f"\n[Run {config_counter}/{total_configs}] Repeat {repeat+1}/{num_repeats}: warmup={w}")

                # set same model initialization seed before building the model (ensures identical initial params)
                set_global_seed(model_seed)
                argan = ARGAN(
                    input_dim=n_features,
                    hidden_dims=hidden_dims,
                    output_dim=output_dim,
                    task_type=task_type,
                    dropout_rate=dropout_rate,
                    warmup_epochs=w,
                    target_keep=1.0 - dropout_rate,
                )

                t0 = time.time()
                argan.train(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    X_val=X_test,
                    y_val=y_test,
                    shuffle_seed=shuffle_seed
                )
                elapsed = time.time() - t0

                # final test eval (ensures eval with no masks/dropout)
                final_loss, final_acc = argan.evaluate(X_test, y_test)

                # collect per-epoch arrays (val arrays will be length == epochs)
                val_acc_arr = np.array(argan.metrics["val_task_accuracy"])
                val_loss_arr = np.array(argan.metrics["val_task_loss"])
                sparsity_arr = np.array(argan.metrics["sparsity"])

                results_per_warmup[w]["val_acc"].append(val_acc_arr)
                results_per_warmup[w]["val_loss"].append(val_loss_arr)
                results_per_warmup[w]["sparsity"].append(sparsity_arr)
                results_per_warmup[w]["final_acc"].append(float(final_acc))
                results_per_warmup[w]["time_s"].append(float(elapsed))

                # small CSV row for this run
                csv_rows.append({
                    "config": config_name,
                    "n_samples": n_samples,
                    "hidden_dims": "-".join(map(str,hidden_dims)),
                    "warmup": w,
                    "repeat": repeat,
                    "final_val_acc": float(final_acc),
                    "final_val_loss": float(final_loss),
                    "time_s": float(elapsed),
                })

                # (Important) clear metrics on argan so metrics don't accumulate across runs if reused
                # Note: we created new argan each time, but defensive zeroing:
                argan.metrics = {k: [] for k in argan.metrics.keys()}

        # End repeats for a given config -> aggregate & plot
        # compute means over repeats
        os.makedirs(out_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"AR-GAN self-benchmark: {config_name}", fontsize=14, fontweight="bold")

        ax_acc = axes[0, 0]
        ax_loss = axes[0, 1]
        ax_spars = axes[1, 0]
        ax_bar = axes[1, 1]

        # colors cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, w in enumerate(warmup_grid):
            runs_acc = results_per_warmup[w]["val_acc"]  # list of arrays
            runs_loss = results_per_warmup[w]["val_loss"]
            runs_spars = results_per_warmup[w]["sparsity"]

            # stack -> shape (repeats, epochs)
            acc_stack = np.stack(runs_acc, axis=0)
            loss_stack = np.stack(runs_loss, axis=0)
            spars_stack = np.stack(runs_spars, axis=0)

            mean_acc = acc_stack.mean(axis=0)
            std_acc = acc_stack.std(axis=0)
            mean_loss = loss_stack.mean(axis=0)
            std_loss = loss_stack.std(axis=0)
            mean_spars = spars_stack.mean(axis=0)
            std_spars = spars_stack.std(axis=0)

            epochs_range = np.arange(len(mean_acc))

            label = f"warmup={w}"
            c = colors[i % len(colors)]
            ax_acc.plot(epochs_range, mean_acc, label=label, color=c, linewidth=2)
            ax_acc.fill_between(epochs_range, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2, color=c)

            ax_loss.plot(epochs_range, mean_loss, label=label, color=c, linewidth=2)
            ax_loss.fill_between(epochs_range, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2, color=c)

            # sparsity: only meaningful if adversarial phase is used; however we plot for both
            ax_spars.plot(epochs_range, mean_spars, label=label, color=c, linewidth=2)
            ax_spars.fill_between(epochs_range, mean_spars - std_spars, mean_spars + std_spars, alpha=0.2, color=c)

            # bar final accuracies (mean across repeats)
            final_mean = np.mean(results_per_warmup[w]["final_acc"])
            ax_bar.bar(label, final_mean, alpha=0.8, color=c, edgecolor='black')

        ax_acc.set_title("Validation Accuracy (mean ± std over repeats)")
        ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy"); ax_acc.grid(True); ax_acc.legend()

        ax_loss.set_title("Validation Loss (mean ± std over repeats)")
        ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss"); ax_loss.grid(True); ax_loss.legend()

        ax_spars.set_title("Average Sparsity (fraction dropped; mean ± std)")
        ax_spars.set_xlabel("Epoch"); ax_spars.set_ylabel("Fraction Dropped"); ax_spars.grid(True); ax_spars.legend()

        ax_bar.set_title("Final test accuracy (mean over repeats)")
        for bar in ax_bar.patches:
            h = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:.3f}", ha='center', va='bottom', fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out_file = os.path.join(out_dir, f"{config_name}_summary.png")
        plt.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved results figure: {out_file}")

    # Save CSV summary
    csv_path = os.path.join(out_dir, "grid_results_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else ["config"])
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)
    print(f"Saved CSV summary: {csv_path}")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Example grid (you can edit these lists)
    GRID_n_samples = [2000,]
    GRID_hidden_dims = [[6, 4, 2],]
    GRID_epochs = 80
    GRID_batch = 32
    GRID_warmups = [GRID_epochs, 20]  # warmup==epochs => no adversarial; 40 => adversarial starts at epoch 40
    GRID_repeats = 3  # repeats per config (mean ± std)

    # run
    start = time.time()
    run_grid_benchmark(
        task_type="classification",
        n_samples_grid=GRID_n_samples,
        n_features=20,
        hidden_dims_grid=GRID_hidden_dims,
        dropout_rate=0.5,
        epochs=GRID_epochs,
        batch_size=GRID_batch,
        warmup_grid=GRID_warmups,
        num_repeats=GRID_repeats,
        out_dir="argan_grid_results",
    )
    print("\nALL DONE in {:.2f}s".format(time.time() - start))
