import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from .lorentz_metric import normsq4, dot4

def metrics(predict, targets, loss_fn, prefix, logger=None):
    """
    This generates metrics reported at the end of each epoch and during validation/testing, as well as the logstring printed to the logger.
    """    
    # if len(targets.shape)==2:
    #     targets = targets.unsqueeze(1)
    loss = loss_fn(predict,targets).numpy()
    angle = AngleDeviation(predict, targets).numpy()
    drsigma = dRSigma(predict, targets).numpy()
    pTsigma = pTSigma(predict, targets).numpy()
    massdelta = MassSigma(predict, targets).numpy()
    loss_inv = loss_fn_inv(predict, targets).numpy()
    loss_m = loss_fn_m(predict, targets).numpy()
    loss_m2 = loss_fn_m2(predict, targets).numpy()
    loss_3d = loss_fn_3d(predict, targets).numpy()
    loss_4d = loss_fn_4d(predict, targets).numpy()
    loss_E = loss_fn_E(predict,targets).numpy()
    loss_psi = loss_fn_psi(predict,targets).numpy()
    loss_pT = loss_fn_pT(predict,targets).numpy()
    loss_dR = loss_fn_dR(predict, targets).numpy()
    loss_col = loss_fn_col(predict,targets).numpy()
    loss_col3 = loss_fn_col3(predict,targets).numpy()
    
    w = 1 + 8 * targets.shape[1]

    metrics = {'loss': loss, '∆Ψ': angle, '∆R': drsigma, '∆pT': pTsigma, '∆m': massdelta, 'loss_inv': loss_inv, 'loss_m': loss_m, 'loss_m2': loss_m2, 'loss_3d': loss_3d, 'loss_4d': loss_4d, "loss_E": loss_E, "loss_psi": loss_psi, "loss_pT": loss_pT, "loss_R": loss_dR, "loss_col": loss_col, "loss_col3": loss_col3}
    with np.printoptions(precision=4):
        f = lambda s: f'{s.item():10.4f}' if s.size==1 else f'{str(s):>{w}}'
        string = f' L: {loss:10.4f}, ∆Ψ: {f(angle)}, ∆R: {f(drsigma)}, ∆pT: {f(pTsigma)}, ∆m: {f(massdelta)}, loss_inv: {loss_inv:10.4f}, loss_m: {loss_m:10.4f}, loss_m2: {loss_m2:10.4f}, loss_3d: {loss_3d:10.4f}, loss_4d: {loss_4d:10.4f}, loss_E: {loss_E:10.4f}, loss_psi: {loss_psi:10.4f}, loss_pT: {loss_pT:10.4f}, loss_R: {loss_dR:10.4f}, loss_col: {loss_col:10.4f}, loss_col3: {loss_col3:10.4f}'
    return metrics, string

def minibatch_metrics(predict, targets, loss):
    """
    This computes metrics for each minibatch (if verbose mode is used). The logstring is defined separately in minibatch_metrics_string.
    """    
    # if len(targets.shape)==2:
    #     targets = targets.unsqueeze(1)
    angle = AngleDeviation(predict, targets).numpy()
    drsigma = dRSigma(predict, targets).numpy()
    pTsigma = pTSigma(predict, targets).numpy()
    massdelta = MassSigma(predict, targets).numpy()
    
    return [loss, angle, drsigma, pTsigma, massdelta]

def minibatch_metrics_string(metrics):
    L, psi, dr, pT, m = metrics
    with np.printoptions(precision=4):
        f = lambda s: f'{s.item():10.4f}' if s.size==1 else f'{str(s):>25}'
        string = f'   L: {L:<12.4f}, ∆Ψ: {f(psi)}, ∆R: {f(dr)}, ∆pT: {f(pT)}, ∆m: {f(m)}'
    return string




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

def AngleDeviation(predict, targets):
    """
    Measures the (always positive) angle between any two 3D vectors and returns the 68% quantile over the batch
    """
    angles = Angle3D(predict[...,1:4], targets[...,1:4])
    return  torch.quantile(angles, 0.68, dim=0)

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
    rel = ((normsq4(predict).abs().sqrt()-normsq4(targets).abs().sqrt())/normsq4(targets).abs().sqrt())
    return iqr(rel, dim=0)  # mass relative error

def pTSigma(predict, targets):
    """
     half of the 68% interquantile range of relative deviation in pT
    """
    rel = ((predict[...,[1,2]].norm(dim=-1)-targets[...,[1,2]].norm(dim=-1))/targets[...,[1,2]].norm(dim=-1))
    return iqr(rel, dim=0)  # pT relative error

def dRSigma(predict, targets):
    return torch.quantile(dR(predict, targets), 0.68, dim=0)

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
    return dR(predict,targets).mean()

def loss_fn_pT(predict,targets):
    return (predict[...,1:3].norm(dim=-1) - targets[...,1:3].norm(dim=-1)).abs().mean()

def mass(x):
    norm=normsq4(x)
    return norm.sign() * (norm.abs()+1e-8).sqrt()


def iqr(x, rng=(0.16, 0.84), dim=0):
    rng = sorted(rng)
    return ((torch.quantile(x,rng[1], dim=dim) - torch.quantile(x,rng[0],dim=dim)) * 0.5)