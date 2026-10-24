"""Regression metrics and losses for predicted 4-vectors (carried over from the original CARDINAL/PELICAN code)."""
import torch
import numpy as np

from ..lorentz import dot4, normsq4


def cart2cyl(cart, include_r=False):
    """ 
    4D Cartesian coordinates to 2D detector coordinates conversion.
	"""
    cart = cart[...,1:4]
    r = cart.norm(dim=-1)
    theta = (cart[..., 2] / r).acos().nan_to_num() # theta=acos(z/r)
    eta = - (theta / 2).tan().log()                # eta=-log(tan(theta/2))
    phi = torch.atan2(cart[..., 1], cart[..., 0])  # phi=atan(y/x)
    if include_r:
        sph = torch.stack((eta, phi, r), dim=-1)
    else:
        sph = torch.stack((eta, phi), dim=-1)
    return sph

def masked_invalid(x, targets):
    """
    Sets to nan the entries of x for events whose target has no 3-momentum (e.g. background events with a zero target),
    where angles and relative deviations are undefined, so that nan-aware quantiles skip them
    """
    return x.masked_fill(targets[...,1:4].norm(dim=-1) == 0, torch.nan)

def AngleDeviation(predict, targets):
    """
    Measures the (always positive) angle between any two 3D vectors and returns the 68% quantile over the batch
    """
    angles = masked_invalid(Angle3D(predict[...,1:4], targets[...,1:4]), targets)
    return  torch.nanquantile(angles, 0.68, dim=0)

def PhiSigma(predict, targets):
    """
    Measures the oriented angle between any two 2D vectors and returns  half of the 68% interquantile range over the batch
    """
    angles = Angle2D(predict[...,1:3], targets[...,1:3])
    return  iqr(angles, dim=0)

def Angle2D(u, v):
    """
    Measures the oriented angle between any two 2D vectors (allows batches)
    """
    dots = (u * v).sum(dim=-1)
    j = torch.tensor([[0,1],[-1,0]], device=u.device, dtype=u.dtype)
    dets = (u * torch.einsum("ab,...b->...a", j, v)).sum(dim=-1)
    angles = torch.atan2(dets, dots)
    return angles

def Angle3D(u, v):
    """
    Measures the (always positive) angle between any two 3D vectors (allows batches)
    """
    aux1 = u.norm(dim=-1).unsqueeze(-1) * v
    aux2 = v.norm(dim=-1).unsqueeze(-1) * u
    angles = 2*torch.atan2((aux1 - aux2).norm(dim=-1), (aux1 + aux2).norm(dim=-1))
    return angles

def dR(u,v):
    """
    Measures the R between two vectors, defined as dR^2=d(phi)^2 + d(theta)^2/sin(theta)^2,
    """
    return (cart2cyl(u)-cart2cyl(v)).norm(dim=-1)

def MassSigma(predict, targets):
    """
    half of the 68% interquantile range over of relative deviation in mass
    """
    rel = masked_invalid((normsq4(predict).abs().sqrt()-normsq4(targets).abs().sqrt())/normsq4(targets).abs().sqrt(), targets)
    return iqr(rel, dim=0)  # mass relative error

def pTSigma(predict, targets):
    """
     half of the 68% interquantile range of relative deviation in pT
    """
    rel = masked_invalid((predict[...,[1,2]].norm(dim=-1)-targets[...,[1,2]].norm(dim=-1))/targets[...,[1,2]].norm(dim=-1), targets)
    return iqr(rel, dim=0)  # pT relative error

def dRSigma(predict, targets):
    return torch.nanquantile(masked_invalid(dR(predict, targets), targets), 0.68, dim=0)

def loss_fn_col(predict, targets):
    return (dot4(predict,targets)**2 - dot4(predict,predict)*dot4(targets,targets) + 1e-6).abs().pow(0.5).mean()

def loss_fn_col3(predict, targets):
    u = predict[...,1:4].norm(dim=-1)
    v = targets[...,1:4].norm(dim=-1)
    uv = (predict[...,1:4]*targets[...,1:4]).sum(dim=-1)
    return (u*v - uv).abs().mean()

def loss_fn_inv(predict, targets):
    return normsq4(predict - targets).abs().mean()

def loss_fn_m(predict, targets):
    return (mass(predict) - mass(targets)).abs().mean()

def loss_fn_m2(predict, targets):
    return (normsq4(predict) - normsq4(targets)).abs().mean()

def loss_fn_3d(predict, targets):
    return ((predict[...,1:4] - targets[...,1:4]).norm(dim=-1)).mean()

def loss_fn_4d(predict, targets):
    return (predict-targets).norm(dim=-1).mean()

def loss_fn_E(predict,targets):
    return (predict[...,0]-targets[...,0]).abs().mean()

def loss_fn_psi(predict,targets):
    return Angle3D(predict[...,1:4], targets[...,1:4]).mean()

def loss_fn_dR(predict,targets):
    return masked_invalid(dR(predict,targets), targets).nanmean()

def loss_fn_pT(predict,targets):
    return (predict[...,1:3].norm(dim=-1) - targets[...,1:3].norm(dim=-1)).abs().mean()

def mass(x):
    norm=normsq4(x)
    return norm.sign() * (norm.abs()+1e-8).sqrt()


def iqr(x, rng=(0.16, 0.84), dim=0):
    rng = sorted(rng)
    return ((torch.nanquantile(x,rng[1], dim=dim) - torch.nanquantile(x,rng[0],dim=dim)) * 0.5)
