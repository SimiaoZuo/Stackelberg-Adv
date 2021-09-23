import higher
import logging
import math
import torch
import torch.nn as nn

from fairseq import metrics
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss

logger = logging.getLogger(__name__)


def reduce_adv_unroll_metrics(logging_outputs):
    """Aggregate logging outputs from data parallel training."""
    adv_loss_sum = sum(log.get('adv_loss', 0) for log in logging_outputs)
    ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
    metrics.log_scalar('adv_loss', adv_loss_sum / ntokens / math.log(2), ntokens, round=3)


def KL_loss(p, q, pad_mask):
    """ symmetric KL-divergence 1/2*(KL(p||q)+KL(q||p)) """
    p, q = p.float(), q.float()
    dict_size = q.size(-1)
    non_pad_mask = ~pad_mask
    p = p.view(-1, dict_size)[non_pad_mask]
    q = q.view(-1, dict_size)[non_pad_mask]
    loss = (p - q) * (torch.log(p) - torch.log(q))
    return 0.5 * loss.sum()


def forward_loss(input_loss):
    """ Forward pass to get the loss. """

    model = input_loss['model']
    decoder_out = input_loss['decoder_out']
    padding_idx = input_loss['padding_idx']
    target = input_loss['target']
    tgt_pad_mask = input_loss['tgt_pad_mask']
    p = input_loss['p']

    # virtual adversarial training loss
    def get_vat_loss():
        q = model.get_normalized_probs(decoder_out, log_probs=False)
        result = KL_loss(p, q, tgt_pad_mask)
        return result

    # adversarial training loss
    def get_adv_loss():
        log_prob = model.get_normalized_probs(decoder_out, log_probs=True)
        log_prob = log_prob.view(-1, log_prob.size(-1))
        result, _ = label_smoothed_nll_loss(
            log_prob, target.view(-1, 1), model.args.label_smoothing,
            ignore_index=padding_idx, reduce=True,
        )
        return result

    if model.args.use_reg == 1:  # VAT
        loss = get_vat_loss()
    elif model.args.use_reg == 2:  # Adv
        loss = get_adv_loss()
    else:
        raise NotImplementedError('Invalid use_reg argument.')

    return loss


def perturb_unroll(input_unroll):
    model = input_unroll['model']
    sample = input_unroll['sample']
    adv_eps = input_unroll['adv_eps']
    padding_idx = input_unroll['padding_idx']
    target = input_unroll['target']
    tgt_pad_mask = input_unroll['tgt_pad_mask']
    p = input_unroll['p']
    loss_scale = input_unroll['loss_scale']
    perturbation_enc = input_unroll['perturbation_enc']
    perturbation_dec = input_unroll['perturbation_dec']

    # pre-processing, set the initial perturbation
    # the "Higher" package requires all the parameters inside the model
    assert model.encoder.perturbation_enc is None and model.decoder.perturbation_dec is None
    params = []
    if 0 in model.args.perturbation_target:
        model.encoder.perturbation_enc = nn.Parameter(perturbation_enc, requires_grad=True)
        params.append(model.encoder.perturbation_enc)
    if 1 in model.args.perturbation_target:
        model.decoder.perturbation_dec = nn.Parameter(perturbation_dec, requires_grad=True)
        params.append(model.decoder.perturbation_dec)

    opt = torch.optim.SGD(params, lr=model.args.inner_lr)
    num_unroll = model.args.inner_steps
    with higher.innerloop_ctx(model, opt, copy_initial_weights=True) as (fmodel, diffopt):
        # gradient ascent steps
        for _ in range(num_unroll):
            decoder_out = fmodel(**sample['net_input'], adv_step=1)
            input_loss = {
                'model': fmodel,
                'decoder_out': decoder_out,
                'padding_idx': padding_idx,
                'target': target,
                'tgt_pad_mask': tgt_pad_mask,
                'p': p,
            }
            loss = forward_loss(input_loss) * -1.0
            if loss_scale is not None:
                loss = loss * loss_scale

            diffopt.step(loss)

        # this is used to compute the leader term
        perturbation_enc_save = torch.renorm(model.encoder.perturbation_enc.detach().clone(),
                                             p=2, dim=0, maxnorm=adv_eps)
        perturbation_dec_save = torch.renorm(model.decoder.perturbation_dec.detach().clone(),
                                             p=2, dim=0, maxnorm=adv_eps)

        # unroll to compute Stackelberg gradients
        decoder_out = fmodel(**sample['net_input'], adv_step=1)
        input_loss = {
            'model': fmodel,
            'decoder_out': decoder_out,
            'padding_idx': padding_idx,
            'target': target,
            'tgt_pad_mask': tgt_pad_mask,
            'p': p,
        }
        loss = forward_loss(input_loss)
        if loss_scale is not None:
            loss = loss * loss_scale

        if 0 in model.args.perturbation_target:
            fmodel.encoder.perturbation_enc.retain_grad()
        if 1 in model.args.perturbation_target:
            fmodel.decoder.perturbation_dec.retain_grad()
        loss.backward()

        # copy the Stackelberg gradients from the differentiable model to the one we use
        param_dict = {}
        for name, param in fmodel.named_parameters():
            param_dict[name] = param.grad.detach().clone()
        for name, param in model.named_parameters():
            param.grad = param_dict[name]

    # we don't need the perturbations anymore
    model.encoder.perturbation_enc = None
    model.decoder.perturbation_dec = None

    return perturbation_enc_save, perturbation_dec_save


def compute_unroll_loss(model, sample, match_output, padding_idx=-1, loss_scale=None):
    target = model.get_targets(sample, match_output).view(-1)
    tgt_pad_mask = target.eq(padding_idx)
    p = model.get_normalized_probs(match_output, log_probs=False).detach()  # do not require grad
    adv_eps = model.args.eps
    src_tokens = sample['net_input']['src_tokens']
    prev_output_tokens = sample['net_input']['prev_output_tokens']

    embed_weights = model.encoder.embed_tokens.weight
    embed_size = embed_weights.size(-1)
    dtype = embed_weights.dtype
    device = embed_weights.device

    def generate_perturbation():
        # random initialization
        enc, dec = None, None
        if 0 in model.args.perturbation_target:
            temp = src_tokens.size()
            enc = torch.normal(mean=0.0, std=model.args.init_std,
                               size=[temp[0], temp[1], embed_size], dtype=dtype, device=device)
        if 1 in model.args.perturbation_target:
            temp = prev_output_tokens.size()
            dec = torch.normal(mean=0.0, std=model.args.init_std,
                               size=[temp[0], temp[1], embed_size], dtype=dtype, device=device)
        return enc, dec

    perturbation_enc, perturbation_dec = generate_perturbation()

    input_unroll = {
        'model': model,
        'sample': sample,
        'adv_eps': adv_eps,
        'padding_idx': padding_idx,
        'target': target,
        'tgt_pad_mask': tgt_pad_mask,
        'p': p,
        'loss_scale': loss_scale,
        'perturbation_enc': perturbation_enc,
        'perturbation_dec': perturbation_dec,
    }
    save_enc, save_dec = perturb_unroll(input_unroll)

    # this is used to compute the leader gradients, i.e., from the standard VAT loss.
    decoder_out = model(**sample['net_input'], perturbation_enc=save_enc,
                        perturbation_dec=save_dec, adv_step=1)
    input_loss = {
        'model': model,
        'decoder_out': decoder_out,
        'padding_idx': padding_idx,
        'target': target,
        'tgt_pad_mask': tgt_pad_mask,
        'p': p,
    }
    loss = forward_loss(input_loss)

    return loss
