import torch

def spearman_correlation_loss(preds, targets):
    pred_rank = preds.argsort(dim=1).argsort(dim=1).float()
    target_rank = targets.argsort(dim=1).argsort(dim=1).float()

    cov = ((pred_rank - pred_rank.mean(dim=1, keepdim=True)) *
           (target_rank - target_rank.mean(dim=1, keepdim=True))).mean(dim=1)
    std_pred = pred_rank.std(dim=1)
    std_target = target_rank.std(dim=1)

    return 1 - (cov / (std_pred * std_target + 1e-8)).mean()