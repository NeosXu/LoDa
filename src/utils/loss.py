import torch


def plcc_loss(y_pred, y):
    y = y.detach().float()
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()


def get_loss(cfg):
    loss_fn = []
    for name, weight in cfg.loss.fn:
        if name == "plcc_loss":
            fn = plcc_loss
        else:
            raise Exception("%s loss not supported" % name)

        loss_fn.append((fn, weight))

    return loss_fn
