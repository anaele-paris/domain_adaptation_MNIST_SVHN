import torch

def cycle_consistency_loss(real_images, reconstructed_images):
    """
    Compute cycle consistency loss.
    
    Args:
        real_images (torch.Tensor): Real images.
        reconstructed_images (torch.Tensor): Reconstructed images.

    Returns:
        torch.Tensor: Cycle consistency loss.
    """
    loss = torch.mean(torch.abs(real_images - reconstructed_images))
    return loss

def identity_loss(real_images, generated_images):
    """
    Compute identity loss.

    Args:
        real_images (torch.Tensor): Real images.
        generated_images (torch.Tensor): Generated images.

    Returns:
        torch.Tensor: Identity loss.
    """
    loss = torch.mean(torch.abs(real_images - generated_images))
    return loss
