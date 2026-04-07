import torch


def multilabel_categorical_crossentropy(y_true, y_pred):
    if y_true.numel() == 0 or y_pred.numel() == 0:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

    loss_mask = y_true != -100

    if not loss_mask.any():
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

    if y_true.size(0) != y_pred.size(0):
        min_size = min(y_true.size(0), y_pred.size(0))
        y_true = y_true[:min_size]
        y_pred = y_pred[:min_size]
        loss_mask = loss_mask[:min_size]

    y_true_masked = y_true.masked_select(loss_mask).view(-1, y_pred.size(-1))
    y_pred_masked = y_pred.masked_select(loss_mask.unsqueeze(-1).expand_as(y_pred)).view(-1, y_pred.size(-1))

    if y_true_masked.numel() == 0 or y_pred_masked.numel() == 0:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

    y_pred_masked = (1 - 2 * y_true_masked) * y_pred_masked
    y_pred_neg = y_pred_masked - y_true_masked * 1e12
    y_pred_pos = y_pred_masked - (1 - y_true_masked) * 1e12
    zeros = torch.zeros_like(y_pred_masked[:, :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()
