"""Shared LSTM and directed perturbation-based Granger-style graph estimation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

HISTORY = 20
EPSILON = 1e-8


class SharedLSTMGC(nn.Module):
    def __init__(self, num_features: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers=2,
                            batch_first=True)
        self.output = nn.Linear(hidden_size, num_features)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.output(self.lstm(history)[0][:, -1])


def _sample_batch(windows: torch.Tensor, time_index: int):
    return (windows[:, :, time_index - HISTORY:time_index].transpose(1, 2),
            windows[:, :, time_index])


def train_predictor(train_data, val_data, device: torch.device, epochs: int = 20,
                    batch_size: int = 64, seed: int = 42) -> SharedLSTMGC:
    """Fit only training windows; use validation windows only for model selection."""
    torch.manual_seed(seed)
    model = SharedLSTMGC(train_data.temporal.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = float("inf"), None
    generator = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        for _, windows, _ in DataLoader(train_data, batch_size=batch_size, shuffle=True):
            windows = windows.to(device)
            time_index = int(generator.integers(HISTORY, windows.shape[-1]))
            history, target = _sample_batch(windows, time_index)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(history), target)
            loss.backward()
            optimizer.step()
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for _, windows, _ in DataLoader(val_data, batch_size=batch_size):
                windows = windows.to(device)
                # A fixed validation position makes epoch comparisons deterministic.
                history, target = _sample_batch(windows, HISTORY)
                total += nn.functional.mse_loss(model(history), target, reduction="sum").item()
                count += target.numel()
        score = total / count
        if score < best:
            best = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"LSTM epoch {epoch + 1}/{epochs}: validation MSE={score:.6f}")
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def estimate_graphs(model: SharedLSTMGC, dataset, prior: np.ndarray,
                    device: torch.device, top_k: int = 5, seed: int = 42,
                    prediction_batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Return per-window source->target scores and binary top-k directed graphs.

    Each window is evaluated independently. The predictor and all non-perturbed
    channels are held fixed. Gaussian replacements use a reproducible RNG.
    """
    n, d, length = dataset.temporal.shape
    if prior.shape != (d, d) or length <= HISTORY:
        raise ValueError("Temporal prior or window shape is incompatible with the LSTM")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    model.eval()
    scores = np.zeros((n, d, d), dtype=np.float32)
    graphs = np.zeros((n, d, d), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    candidate_sources = np.flatnonzero(prior.any(axis=1))
    for window_index in range(n):
        window = dataset.temporal[window_index].to(device).T  # [time, feature]
        histories = window.unfold(0, HISTORY + 1, 1).permute(0, 2, 1)
        inputs, targets = histories[:, :HISTORY], histories[:, HISTORY]

        def error(sequence):
            predictions = []
            for chunk in sequence.split(prediction_batch_size):
                predictions.append(model(chunk))
            return (torch.cat(predictions) - targets).square().mean(0)

        full_error = error(inputs)
        for source in candidate_sources:
            perturbed = inputs.clone()
            noise = rng.standard_normal((len(inputs), HISTORY)).astype(np.float32)
            perturbed[:, :, source] = torch.from_numpy(noise).to(device)
            changed_error = error(perturbed)
            score = torch.clamp(torch.log((changed_error + EPSILON) /
                                          (full_error + EPSILON)), min=0)
            scores[window_index, source] = score.cpu().numpy() * prior[source]
            scores[window_index, source, source] = 0
            eligible = np.flatnonzero(scores[window_index, source] > 0)
            order = sorted(eligible, key=lambda target: (-scores[window_index, source, target], target))
            graphs[window_index, source, order[:top_k]] = 1
        if (window_index + 1) % 500 == 0:
            print(f"Estimated temporal graphs: {window_index + 1}/{n}")
    return scores, graphs
