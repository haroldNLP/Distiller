from Distiller.textbrewer.losses import mi_loss, interpolated_lower_bound
from Distiller.utils import mlp_critic
from Distiller.transformers import AdamW, get_linear_schedule_with_warmup, WEIGHTS_NAME
import torch
from tqdm import tqdm


def sample_fn(rho=0.5, dim=512, batch_size=32):
    x, eps = torch.chunk(torch.normal(0.0, 1.0, size=(batch_size, 2 * dim)), chunks=2, dim=1)
    y = rho * x + torch.tensor(1. - rho ** 2).type(torch.FloatTensor) * eps
    return x, y


import numpy as np


def rho_to_mi(dim, rho):
    return -0.5 * np.log(1 - rho ** 2) * dim


def mi_to_rho(dim, mi):
    return np.sqrt(1 - np.exp(-2.0 / dim * mi))


def mi_schedule(n_iter):
    """Generate schedule for increasing correlation over time."""
    mis = np.round(np.linspace(0.5, 5.5 - 1e-9, n_iter)) * 2.0  # 0.1
    return mis.astype(np.float32)


# Smooting span for Exponential Moving Average
EMA_SPAN = 200
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 3))
for i, rho in enumerate([0.5, 0.99]):
    plt.subplot(1, 2, i + 1)
    x, y = sample_fn(batch_size=500, dim=1, rho=rho)
    plt.scatter(x[:, 0], y[:, 0])
    plt.title(r'$\rho=%.2f$,  $I(X; Y)=%.1f$' % (rho, rho_to_mi(1, rho)))
    plt.xlim(-3, 3);
    plt.ylim(-3, 3)
    plt.savefig("samples.png")
data_params = {
    'dim': 20,
    'batch_size': 64,
}

critic_params = {
    'layers': 2,
    'embed_dim': 32,
    'hidden_dim': 256,
    'activation': 'relu',
}
opt_params = {
    'iterations': 20000,
    'learning_rate': 5e-4,
}
estimators = {}


# Add interpolated bounds
def sigmoid(x):
    return 1 / (1. + np.exp(-x))


for alpha_logit in [-5., 0., 5.]:
    name = 'alpha=%.2f' % sigmoid(alpha_logit)
    estimators[name] = dict(estimator='interpolated',
                            alpha_logit=alpha_logit, baseline='unnormalized')
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_estimator(critic_params, data_params, mi_params, alpha):
    """Main training loop that estimates time-varying MI."""
    # Ground truth rho is only used by conditional critic
    # critic = CRITICS[mi_params.get('critic', 'concat')](rho=None, **critic_params)
    # baseline = BASELINES[mi_params.get('baseline', 'constant')]()

    baseline_fn = mlp_critic(data_params['dim'],
                             hidden_size=512, out_dim=1)
    critic = mlp_critic(data_params['dim'], data_params['dim'], critic_params['hidden_dim'], critic_params['embed_dim'])
    no_decay = ["bias", "LayerNorm.weight", "weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in critic.named_parameters()],
         "weight_decay": 0.0},
        {"params": [p for n, p in baseline_fn.named_parameters()],
         "weight_decay": 0.0}]

    #     for name, param in baseline_fn.named_parameters():
    #         if 'weight' in name:
    #             torch.nn.init.xavier_uniform(param)
    #         elif 'bias' in name:
    #             torch.nn.init.constant_(param, 0)
    optimizer = AdamW(optimizer_grouped_parameters, lr=opt_params['learning_rate'])
    optimizer.zero_grad()

    def train_step(rho, data_params, mi_params):
        # Annoying special case:
        # For the true conditional, the critic depends on the true correlation rho,
        # so we rebuild the critic at each iteration.
        x, y = sample_fn(dim=data_params['dim'], rho=rho, batch_size=data_params['batch_size'])
        log_baseline = torch.squeeze(baseline_fn(y=y))
        scores = critic(x, y)
        mi = interpolated_lower_bound(scores, log_baseline, alpha)
        loss = -mi

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return mi

    # Schedule of correlation over iterations
    mis = mi_schedule(opt_params['iterations'])
    rhos = mi_to_rho(data_params['dim'], mis)

    estimates = []
    for i in tqdm(range(opt_params['iterations'])):
        mi = train_step(rhos[i], data_params, mi_params).detach().numpy()
        if np.isnan(mi):
            print(i)
        estimates.append(mi)

    return np.array(estimates)


if __name__ == "__main__":
    set_seed(916)
    estimates = {}
    for estimator, mi_params in estimators.items():
        print("Training %s..." % estimator)
        estimates[estimator] = train_estimator(critic_params, data_params, mi_params, mi_params['alpha_logit'])

    mi_true = mi_schedule(opt_params['iterations'])
    # Names specifies the key and ordering for plotting estimators
    names = np.sort(list(estimators.keys()))
    lnames = list(map(lambda s: s.replace('alpha', '$\\alpha$'), names))
    nrows = min(2, len(estimates))
    ncols = int(np.ceil(len(estimates) / float(nrows)))
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 3 * nrows))
    if len(estimates) == 1:
        axs = [axs]
    axs = np.ravel(axs)
    import pandas as pd

    for i, name in enumerate(names):
        plt.sca(axs[i])
        plt.title(lnames[i])
        # Plot estimated MI and smoothed MI
        mis = estimates[name]
        mis_smooth = pd.Series(mis).ewm(span=EMA_SPAN).mean()
        p1 = plt.plot(mis, alpha=0.3)[0]
        plt.plot(mis_smooth, c=p1.get_color())
        # Plot true MI and line for log(batch size)
        plt.plot(mi_true, color='k', label='True MI')
        estimator = estimators[name]['estimator']
        if 'interpolated' in estimator or 'nce' in estimator:
            # Add theoretical upper bound lines
            if 'interpolated' in estimator:
                log_alpha = -np.log(1 + np.exp(-estimators[name]['alpha_logit']))
            else:
                log_alpha = 1.
            plt.axhline(1 + np.log(data_params['batch_size']) - log_alpha, c='k', linestyle='--',
                        label=r'1 + log(K/$\alpha$)')
            plt.ylim(-1, mi_true.max() + 1)
            plt.xlim(0, opt_params['iterations'])
        if i == len(estimates) - ncols:
            plt.xlabel('steps')
            plt.ylabel('Mutual information (nats)')
    plt.legend(loc='best', fontsize=8, framealpha=0.0)
    plt.gcf().tight_layout()