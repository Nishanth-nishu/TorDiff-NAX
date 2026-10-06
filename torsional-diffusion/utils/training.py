import time
import numpy as np
from tqdm import tqdm
import torch
import diffusion.torus as torus


def train_epoch(model, loader, optimizer, device, stats=None, fail_on_nan=False, limit_iters=0):
    # [round2] stats (dict, D5 data-wait timer from plan 3), fail_on_nan (D9: NaN-only fail-fast) and limit_iters
    # (D5 measurement) are all default-off; with defaults the numerics are those of the original loop.
    model.train()
    loss_tot = 0
    base_tot = 0
    n_iter, t_wait, t_epoch0 = 0, 0.0, time.time()
    t_ready = time.time()

    for data in tqdm(loader, total=len(loader)):
        t_wait += time.time() - t_ready  # time spent waiting for the loader to deliver this batch
        data = data.to(device)
        optimizer.zero_grad()

        data = model(data)
        pred = data.edge_pred

        score = torus.score(
            data.edge_rotate.cpu().numpy(),
            data.edge_sigma.cpu().numpy())
        score = torch.tensor(score, device=pred.device)
        score_norm = torus.score_norm(data.edge_sigma.cpu().numpy())
        score_norm = torch.tensor(score_norm, device=pred.device)
        loss = ((score - pred) ** 2 / score_norm).mean()

        if fail_on_nan and not np.isfinite(loss.item()):
            raise FloatingPointError(f'non-finite training loss {loss.item()} at iteration {n_iter}')
        loss.backward()
        optimizer.step()
        loss_tot += loss.item()
        base_tot += (score ** 2 / score_norm).mean().item()
        n_iter += 1
        t_ready = time.time()
        if limit_iters and n_iter >= limit_iters:
            break

    n_div = n_iter if limit_iters else len(loader)
    loss_avg = loss_tot / n_div
    base_avg = base_tot / n_div
    if stats is not None:
        stats.update(data_wait=t_wait, epoch_time=time.time() - t_epoch0, n_iter=n_iter)
    return loss_avg, base_avg


@torch.no_grad()
def test_epoch(model, loader, device):
    model.eval()
    loss_tot = 0
    base_tot = 0

    for data in tqdm(loader, total=len(loader)):

        data = data.to(device)
        data = model(data)
        pred = data.edge_pred.cpu()

        score = torus.score(
            data.edge_rotate.cpu().numpy(),
            data.edge_sigma.cpu().numpy())
        score = torch.tensor(score)
        score_norm = torus.score_norm(data.edge_sigma.cpu().numpy())
        score_norm = torch.tensor(score_norm)
        loss = ((score - pred) ** 2 / score_norm).mean()

        loss_tot += loss.item()
        base_tot += (score ** 2 / score_norm).mean().item()

    loss_avg = loss_tot / len(loader)
    base_avg = base_tot / len(loader)
    return loss_avg, base_avg

