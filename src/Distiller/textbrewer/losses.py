import torch.nn.functional as F
import torch
import numpy as np
from typing import List

from .compatibility import mask_dtype

def kd_mse_loss(logits_S, logits_T, temperature=1):
    '''
    Calculate the mse loss between logits_S and logits_T

    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    '''
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    loss = F.mse_loss(beta_logits_S, beta_logits_T)
    return loss


def kd_ce_loss(logits_S, logits_T, temperature=1):
    '''
    Calculate the cross entropy between logits_S and logits_T

    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    '''
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
    return loss


def att_mse_loss(attention_S, attention_T, mask=None):
    '''
    * Calculates the mse loss between `attention_S` and `attention_T`.
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.

    :param torch.Tensor logits_S: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*)
    :param torch.Tensor logits_T: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*)
    :param torch.Tensor mask: tensor of shape  (*batch_size*, *length*)
    '''
    if mask is None:
        attention_S_select = torch.where(attention_S <= -1e-3, torch.zeros_like(attention_S), attention_S)
        attention_T_select = torch.where(attention_T <= -1e-3, torch.zeros_like(attention_T), attention_T)
        loss = F.mse_loss(attention_S_select, attention_T_select)
    else:
        mask = mask.to(attention_S).unsqueeze(1).expand(-1, attention_S.size(1), -1) # (bs, num_of_heads, len)
        valid_count = torch.pow(mask.sum(dim=2),2).sum()
        loss = (F.mse_loss(attention_S, attention_T, reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(2)).sum() / valid_count
    return loss


def att_mse_sum_loss(attention_S, attention_T, mask=None):
    '''
    * Calculates the mse loss between `attention_S` and `attention_T`. 
    * If the the shape is (*batch_size*, *num_heads*, *length*, *length*), sums along the `num_heads` dimension and then calcuates the mse loss between the two matrices.
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.

    :param torch.Tensor logits_S: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*) or (*batch_size*, *length*, *length*)
    :param torch.Tensor logits_T: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*) or (*batch_size*, *length*, *length*)
    :param torch.Tensor mask:     tensor of shape  (*batch_size*, *length*)
    '''
    if len(attention_S.size())==4:
        attention_T = attention_T.sum(dim=1)
        attention_S = attention_S.sum(dim=1)
    if mask is None:
        attention_S_select = torch.where(attention_S <= -1e-3, torch.zeros_like(attention_S), attention_S)
        attention_T_select = torch.where(attention_T <= -1e-3, torch.zeros_like(attention_T), attention_T)
        loss = F.mse_loss(attention_S_select, attention_T_select)
    else:
        mask = mask.to(attention_S)
        valid_count = torch.pow(mask.sum(dim=1), 2).sum()
        loss = (F.mse_loss(attention_S, attention_T, reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(1)).sum() / valid_count
    return loss


def att_ce_loss(attention_S, attention_T, mask=None):
    '''

    * Calculates the cross-entropy loss between `attention_S` and `attention_T`, where softmax is to applied on ``dim=-1``.
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.
    
    :param torch.Tensor logits_S: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*)
    :param torch.Tensor logits_T: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*)
    :param torch.Tensor mask:     tensor of shape  (*batch_size*, *length*)
    '''
    probs_T = F.softmax(attention_T, dim=-1)
    if mask is None:
        probs_T_select = torch.where(attention_T <= -1e-3, torch.zeros_like(attention_T), probs_T)
        loss = -((probs_T_select * F.log_softmax(attention_S, dim=-1)).sum(dim=-1)).mean()
    else:
        mask = mask.to(attention_S).unsqueeze(1).expand(-1, attention_S.size(1), -1) # (bs, num_of_heads, len)
        loss = -((probs_T * F.log_softmax(attention_S, dim=-1) * mask.unsqueeze(2)).sum(dim=-1) * mask).sum() / mask.sum()
    return loss


def att_ce_mean_loss(attention_S, attention_T, mask=None):
    '''
    * Calculates the cross-entropy loss between `attention_S` and `attention_T`, where softmax is to applied on ``dim=-1``.
    * If the shape is (*batch_size*, *num_heads*, *length*, *length*), averages over dimension `num_heads` and then computes cross-entropy loss between the two matrics.
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.
    
    :param torch.tensor logits_S: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*) or (*batch_size*, *length*, *length*)
    :param torch.tensor logits_T: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*) or (*batch_size*, *length*, *length*)
    :param torch.tensor mask:     tensor of shape  (*batch_size*, *length*)
    '''
    if len(attention_S.size())==4:
        attention_S = attention_S.mean(dim=1) # (bs, len, len)
        attention_T = attention_T.mean(dim=1)
    probs_T = F.softmax(attention_T, dim=-1)
    if mask is None:
        probs_T_select = torch.where(attention_T <= -1e-3, torch.zeros_like(attention_T), probs_T)
        loss = -((probs_T_select * F.log_softmax(attention_S, dim=-1)).sum(dim=-1)).mean()
    else:
        mask = mask.to(attention_S)
        loss = -((probs_T * F.log_softmax(attention_S, dim=-1) * mask.unsqueeze(1)).sum(dim=-1) * mask).sum() / mask.sum()
    return loss


def hid_ce_loss(state_S, state_T, mask=None):
    probs_T = F.softmax(state_T, dim=-1)
    if mask is None:
        # probs_T_select = torch.where(attention_T <= -1e-3, torch.zeros_like(attention_T), probs_T)
        loss = -((probs_T * F.log_softmax(state_S, dim=-1)).sum(dim=-1)).mean()
    else:
        # mask = mask.to(state_S).unsqueeze(1).expand(-1, attention_S.size(1), -1)  # (bs, num_of_heads, len)
        mask = mask.to(state_S)
        loss = -((probs_T * F.log_softmax(state_S, dim=-1) * mask.unsqueeze(2)).sum(
            dim=-1) * mask).sum() / mask.sum()
    return loss


def hid_mse_loss(state_S, state_T, mask=None):
    '''
    * Calculates the mse loss between `state_S` and `state_T`, which are the hidden state of the models.
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.
    * If the hidden sizes of student and teacher are different, 'proj' option is required in `inetermediate_matches` to match the dimensions.

    :param torch.Tensor state_S: tensor of shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.Tensor state_T: tensor of shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.Tensor mask:    tensor of shape  (*batch_size*, *length*)
    '''
    if mask is None:
        loss = F.mse_loss(state_S, state_T)
    else:
        mask = mask.to(state_S)
        valid_count = mask.sum() * state_S.size(-1)
        loss = (F.mse_loss(state_S, state_T, reduction='none') * mask.unsqueeze(-1)).sum() / valid_count
    return loss


def cos_loss(state_S, state_T, mask=None):
    '''
    * Computes the cosine similarity loss between the inputs. This is the loss used in DistilBERT, see `DistilBERT <https://arxiv.org/abs/1910.01108>`_
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.
    * If the hidden sizes of student and teacher are different, 'proj' option is required in `inetermediate_matches` to match the dimensions.

    :param torch.Tensor state_S: tensor of shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.Tensor state_T: tensor of shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.Tensor mask:    tensor of shape  (*batch_size*, *length*)
    '''
    if mask is  None:
        state_S = state_S.view(-1,state_S.size(-1))
        state_T = state_T.view(-1,state_T.size(-1))
    else:
        mask = mask.to(state_S).unsqueeze(-1).expand_as(state_S).to(mask_dtype) #(bs,len,dim)
        state_S = torch.masked_select(state_S, mask).view(-1, mask.size(-1))  #(bs * select, dim)
        state_T = torch.masked_select(state_T, mask).view(-1, mask.size(-1))  # (bs * select, dim)

    target = state_S.new(state_S.size(0)).fill_(1)
    loss = F.cosine_embedding_loss(state_S, state_T, target, reduction='mean')
    return loss


def pkd_loss(state_S, state_T, mask=None):
    '''
    * Computes normalized vector mse loss at position 0 along `length` dimension. This is the loss used in BERT-PKD, see `Patient Knowledge Distillation for BERT Model Compression <https://arxiv.org/abs/1908.09355>`_.
    * If the hidden sizes of student and teacher are different, 'proj' option is required in `inetermediate_matches` to match the dimensions.
    * If the input tensors are of shape (*batch_size*, *hidden_size*), it directly computes the loss between tensors without taking the hidden states at position 0.

    :param torch.Tensor state_S: tensor of shape  (*batch_size*, *length*, *hidden_size*) or (*batch_size*, *hidden_size*)
    :param torch.Tensor state_T: tensor of shape  (*batch_size*, *length*, *hidden_size*) or (*batch_size*, *hidden_size*)
    :param mask: not used.
    '''
    if state_T.dim()==3:
        cls_T = state_T[:,0] # (batch_size, hidden_dim)
    else:
        cls_T = state_T
    if state_S.dim()==3:
        cls_S = state_S[:,0] # (batch_size, hidden_dim)
    else:
        cls_S = state_S
    normed_cls_T = cls_T/torch.norm(cls_T,dim=1,keepdim=True)
    normed_cls_S = cls_S/torch.norm(cls_S,dim=1,keepdim=True)
    loss = (normed_cls_S - normed_cls_T).pow(2).sum(dim=-1).mean()
    return loss


def fsp_loss(state_S, state_T, mask=None):
    '''
    * Takes in two lists of matrics `state_S` and `state_T`. Each list contains two matrices of the shape (*batch_size*, *length*, *hidden_size*). Computes the similarity matrix between the two matrices in `state_S` ( with the resulting shape (*batch_size*, *hidden_size*, *hidden_size*) ) and the ones in B ( with the resulting shape (*batch_size*, *hidden_size*, *hidden_size*) ), then computes the mse loss between the similarity matrices:

    .. math::

        loss = mean((S_{1}^T \cdot S_{2} - T_{1}^T \cdot T_{2})^2)

    * It is a Variant of FSP loss in `A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning <http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf>`_.
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.
    * If the hidden sizes of student and teacher are different, 'proj' option is required in `inetermediate_matches` to match the dimensions.

    :param torch.tensor state_S: list of two tensors, each tensor is of the shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.tensor state_T: list of two tensors, each tensor is of the shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.tensor mask:    tensor of the shape  (*batch_size*, *length*)

    Example in `intermediate_matches`::

        intermediate_matches = [
        {'layer_T':[0,0], 'layer_S':[0,0], 'feature':'hidden','loss': 'fsp', 'weight' : 1, 'proj':['linear',384,768]},
        ...]
    '''
    if mask is None:
        state_S_0 = state_S[0] # (batch_size , length, hidden_dim)
        state_S_1 = state_S[1] # (batch_size,  length, hidden_dim)
        state_T_0 = state_T[0]
        state_T_1 = state_T[1]
        gram_S = torch.bmm(state_S_0.transpose(1, 2), state_S_1) / state_S_1.size(1)  # (batch_size, hidden_dim, hidden_dim)
        gram_T = torch.bmm(state_T_0.transpose(1, 2), state_T_1) / state_T_1.size(1)
    else:
        mask = mask.to(state_S[0]).unsqueeze(-1)
        lengths = mask.sum(dim=1,keepdim=True)
        state_S_0 = state_S[0] * mask
        state_S_1 = state_S[1] * mask
        state_T_0 = state_T[0] * mask
        state_T_1 = state_T[1] * mask
        gram_S = torch.bmm(state_S_0.transpose(1,2), state_S_1)/lengths
        gram_T = torch.bmm(state_T_0.transpose(1,2), state_T_1)/lengths
    loss = F.mse_loss(gram_S, gram_T)
    return loss


def mmd_loss(state_S, state_T, mask=None):
    '''
    * Takes in two lists of matrices `state_S` and `state_T`. Each list contains 2 matrices of the shape (*batch_size*, *length*, *hidden_size*). `hidden_size` of matrices in `State_S` doesn't need to be the same as that of `state_T`. Computes the similarity matrix between the two matrices in `state_S` ( with the resulting shape (*batch_size*, *length*, *length*) ) and the ones in B ( with the resulting shape (*batch_size*, *length*, *length*) ), then computes the mse loss between the similarity matrices:
    
    .. math::

            loss = mean((S_{1} \cdot S_{2}^T - T_{1} \cdot T_{2}^T)^2)

    * It is a Variant of the NST loss in `Like What You Like: Knowledge Distill via Neuron Selectivity Transfer <https://arxiv.org/abs/1707.01219>`_
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.

    :param torch.tensor state_S: list of two tensors, each tensor is of the shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.tensor state_T: list of two tensors, each tensor is of the shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.tensor mask:    tensor of the shape  (*batch_size*, *length*)

    Example in `intermediate_matches`::

        intermediate_matches = [
        {'layer_T':[0,0], 'layer_S':[0,0], 'feature':'hidden','loss': 'nst', 'weight' : 1},
        ...]
    '''
    state_S_0 = state_S[0] # (batch_size , length, hidden_dim_S)
    state_S_1 = state_S[1] # (batch_size , length, hidden_dim_S)
    state_T_0 = state_T[0] # (batch_size , length, hidden_dim_T)
    state_T_1 = state_T[1] # (batch_size , length, hidden_dim_T)
    if mask is None:
        gram_S = torch.bmm(state_S_0, state_S_1.transpose(1, 2)) / state_S_1.size(2)  # (batch_size, length, length)
        gram_T = torch.bmm(state_T_0, state_T_1.transpose(1, 2)) / state_T_1.size(2)
        loss = F.mse_loss(gram_S, gram_T)
    else:
        mask = mask.to(state_S[0])
        valid_count = torch.pow(mask.sum(dim=1), 2).sum()
        gram_S = torch.bmm(state_S_0, state_S_1.transpose(1, 2)) / state_S_1.size(1)  # (batch_size, length, length)
        gram_T = torch.bmm(state_T_0, state_T_1.transpose(1, 2)) / state_T_1.size(1)
        loss = (F.mse_loss(gram_S, gram_T, reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(1)).sum() / valid_count
    return loss


def mi_loss(state_S, state_T, critic, baseline_fn, alpha, mask_T=None, mask_S=None):
    if state_T.dim() == 3:
        # cls label states
        if critic.type != 'mlp':
            cls_T = state_T
        else:
            cls_T = state_T[:, 0]  # (batch_size, hidden_dim)
        # cls_T = state_T.view(state_T.shape[0],-1)
        # mean pooling
        # cls_T =
    else:
        cls_T = state_T
    if state_S.dim() == 3:
        # cls label states
        if critic.type != 'mlp':
            cls_S = state_S
        else:
            cls_S = state_S[:, 0]  # (batch_size, hidden_dim)
        # cls_S = state_S.view(state_S.shape[0], -1)
    else:
        cls_S = state_S
    log_baseline = torch.squeeze(baseline_fn(y=cls_T, mask_T=mask_T))
    scores = critic(cls_S, cls_T, mask_S=mask_S, mask_T=mask_T)
    return -interpolated_lower_bound(scores, log_baseline, alpha)



def log_prob_gaussian(x):
    return torch.sum(torch.distributions.normal.Normal(0.,1.).log_prob(x), dim=-1, keepdim=False)


def reduce_logmeanexp_nodiag(x, axis=None):
    batch_size = x.shape[0]
    diag_inf = torch.diag(torch.tensor(np.inf) * torch.ones(batch_size)).to(x.device)
    logsumexp = torch.logsumexp(x - diag_inf, dim=(0,1))
    if axis:
        num_elem = batch_size - 1.
    else:
        num_elem = batch_size * (batch_size - 1.)
    return logsumexp - torch.log(torch.tensor(num_elem))


def log_interpolate(log_a, log_b, alpha_logit):
    """Numerically stable implementation of log(alpha * a + (1-alpha) * b)."""
    log_alpha = -torch.nn.functional.softplus(torch.tensor(-alpha_logit))
    log_1_minus_alpha = -torch.nn.functional.softplus(torch.tensor(alpha_logit))
    y = torch.logsumexp(torch.stack((log_alpha + log_a, log_1_minus_alpha + log_b),0), dim=0)
    return y


def softplus_inverse(x):
    # x = x.numpy()
    threshold = torch.log(torch.tensor(torch.finfo(x.dtype).eps)) + torch.tensor(2.)
    is_too_small = x < torch.exp(threshold)
    is_too_large = x > -threshold
    too_small_value = torch.log(x).type(x.dtype)
    too_large_value = x
    # This `where` will ultimately be a NOP because we won't select this
    # codepath whenever we used the surrogate `ones_like`.
    x = torch.where(is_too_small | is_too_large, torch.ones([], dtype=x.dtype).to(x.device), x)
    y = x + torch.log(-torch.expm1(-x)).type(x.dtype)  # == log(expm1(x))
    return torch.where(is_too_small, too_small_value, torch.where(is_too_large, too_large_value, y))
    # return torch.log(torch.exp(x) - torch.tensor(1.))


def compute_log_loomean(scores):
    """Compute the log leave-one-out mean of the exponentiated scores.

    For each column j we compute the log-sum-exp over the row holding out column j.
    This is a numerically stable version of:
    log_loosum = scores + tfp.math.softplus_inverse(tf.reduce_logsumexp(scores, axis=1, keepdims=True) - scores)
    Implementation based on tfp.vi.csiszar_divergence.csiszar_vimco_helper.
    """
    max_scores = torch.max(scores, dim=1, keepdim=True)[0]
    lse_minus_max = torch.logsumexp(scores - max_scores, dim=1, keepdim=True)
    d = lse_minus_max + (max_scores - scores)
    d_ok = torch.ne(d, torch.tensor(0.))
    safe_d = torch.where(d_ok, d, torch.ones_like(d))
    loo_lse = scores + softplus_inverse(safe_d)
    # Normalize to get the leave one out log mean exp
    loo_lme = loo_lse - torch.log(torch.tensor(scores.shape[1] - 1.))
    return loo_lme


def interpolated_lower_bound(scores, baseline, alpha_logit):
    """Interpolated lower bound on mutual information.

    Interpolates between the InfoNCE baseline ( alpha_logit -> -infty),
    and the single-sample TUBA baseline (alpha_logit -> infty)

    Args:
    scores: [batch_size, batch_size] critic scores
    baseline: [batch_size] log baseline scores
    alpha_logit: logit for the mixture probability

    Returns:
    scalar, lower bound on MI
    """
    batch_size = scores.shape[0]
    # Compute InfoNCE baseline
    nce_baseline = compute_log_loomean(scores)
    # Inerpolated baseline interpolates the InfoNCE baseline with a learned baseline
    interpolated_baseline = log_interpolate(
      nce_baseline, torch.tile(baseline[:, None], (1, batch_size)), alpha_logit)
    # Marginal term.
    critic_marg = scores - torch.diagonal(interpolated_baseline, offset=0)[:, None]
    marg_term = torch.exp(reduce_logmeanexp_nodiag(critic_marg))

    # Joint term.
    critic_joint = torch.diagonal(scores, offset=0)[:, None] - interpolated_baseline
    joint_term = (torch.sum(critic_joint) -
                torch.sum(torch.diagonal(critic_joint, offset=0))) / (batch_size * (batch_size - 1.))
    return torch.tensor(1.) + joint_term - marg_term


def infonce_lower_bound(scores):
  """InfoNCE lower bound from van den Oord et al. (2018)."""
  nll = torch.mean(torch.diagonal(scores) - torch.logsumexp(scores, dim=1), dim=-1)
  # Alternative implementation:
  # nll = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=tf.range(batch_size))
  mi = torch.log(scores.shape[0]) + nll
  return mi



def tuba_lower_bound(scores, log_baseline=None):
  if log_baseline is not None:
    scores -= log_baseline[:, None]
  batch_size = scores.shape[0]
  # First term is an expectation over samples from the joint,
  # which are the diagonal elmements of the scores matrix.
  joint_term = torch.mean(torch.diagonal(scores), dim=-1)
  # Second term is an expectation over samples from the marginal,
  # which are the off-diagonal elements of the scores matrix.
  marg_term = torch.exp(reduce_logmeanexp_nodiag(scores))
  return 1. + joint_term - marg_term

def nwj_lower_bound(scores):
  # equivalent to: tuba_lower_bound(scores, log_baseline=1.)
  return tuba_lower_bound(scores - 1.)

def nce_loss(state_S, state_T, mask=None):
    # TO be justified
    criterion_t = ContrastLoss(state_T.shape[0])
    criterion_s = ContrastLoss(state_S.shape[0])
    return criterion_t(state_T) + criterion_s(state_S)


class ContrastLoss(torch.nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1
        eps = 1e-7
        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        P_pos[P_pos == 0] = eps
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss