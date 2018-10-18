import torch
import torch.autograd as autograd
import torch.nn.functional as F

__all__ = ['reduce',
        'minimax_generator_loss', 'minimax_discriminator_loss',
        'wasserstein_generator_loss', 'wasserstein_discriminator_loss', 'wasserstein_gradient_penalty',
        'least_squares_generator_loss', 'least_squares_discriminator_loss']


def reduce(x, reduction=None):
    if reduction == "elementwise_mean":
        return torch.mean(x)
    elif reduction == "sum":
        return torch.sum(x)
    else:
        return x


def minimax_generator_loss(d_gen, nonsaturating=True, reduction='elementwise_mean'):
    if nonsaturating:
        target = torch.ones_like(d_gen)
        return F.binary_cross_entropy_with_logits(d_gen, target,
                                                  reduction=reduction)
    else:
        target = torch.zeros_like(d_gen)
        return -1.0 * F.binary_cross_entropy_with_logits(d_gen, target,
                                                         reduction=reduction)


def minimax_discriminator_loss(d_real, d_gen, reduction='elementwise_mean'):
    target_ones = torch.ones_like(d_gen)
    target_zeros = torch.zeros_like(d_real)
    loss = F.binary_cross_entropy_with_logits(d_real, target_ones,
                                              reduction=reduction)
    loss += F.binary_cross_entropy_with_logits(d_gen, target_zeros,
                                               reduction=reduction)
    return loss


def wasserstein_generator_loss(f_gen, reduction='elementwise_mean'):
    return reduce(-1.0 * f_gen, reduction)


def wasserstein_discriminator_loss(f_sample, f_gen, reduction='elementwise_mean'):
    return reduce(f_gen - f_sample, reduction)


def wasserstein_gradient_penalty(interpolate, d_interpolate, reduction='elementwise_mean'):
    grad_outputs = torch.ones_like(d_interpolate)
    gradients = autograd.grad(outputs=d_interpolate, inputs=interpolate,
                              grad_outputs=grad_outputs,
                              create_graph=True, retain_graph=True,
                              only_inputs=True)[0]

    gradient_penalty = (gradients.norm(2) - 1) ** 2
    return reduce(gradient_penalty, reduction)


def least_squares_generator_loss(d_gen, c=1.0, reduction='elementwise_mean'):
    return 0.5 * reduce((d_gen - c) ** 2, reduction)


def least_squares_discriminator_loss(d_real, d_gen, a=0.0, b=1.0, reduction='elementwise_mean'):
    return 0.5 * (reduce((d_real - b) ** 2, reduction) + reduce((d_gen - a) ** 2, reduction))
