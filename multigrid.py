import torch
import torch.nn as nn
from dataclasses import dataclass
from utils import interpolate, restrict, laplace_smoothen

@dataclass
class BoundaryCondition:
    mask: torch.Tensor
    vals: torch.Tensor

    def restrict(self) -> BoundaryCondition:
        return BoundaryCondition(restrict(self.mask), restrict(self.vals))

def laplace_jacobi_step(f: torch.Tensor, bc: BoundaryCondition):
    return torch.where(bc.mask, bc.vals, laplace_smoothen(f))

def laplace_jacobi_smoothen(f: torch.Tensor, bc: BoundaryCondition, tol=1e-3, maxiter=200) -> torch.Tensor:
    for _ in range(maxiter):
        f_smooth = laplace_jacobi_step(f, bc)
        if torch.mean(torch.abs(f - f_smooth)) <= tol:
            return f_smooth
        f = f_smooth
    return f

### Laplace Solvers ###

# V-Cycle
def laplace_multigrid_v_cycle(f: torch.Tensor, bc: BoundaryCondition, depth=4, tol=1e-5, maxiter=200) -> torch.Tensor:
    if not depth:
        return laplace_jacobi_smoothen(f, bc, tol, maxiter)
    f = laplace_jacobi_smoothen(f, bc, tol, maxiter)
    f = interpolate(laplace_multigrid_v_cycle(restrict(f), bc.restrict(), depth-1, tol, maxiter))
    f = laplace_jacobi_smoothen(f, bc, tol, maxiter)
    return f

# F-Cycle
def laplace_multigrid_f_cycle(f: torch.Tensor, bc: BoundaryCondition, depth=4, tol=1e-5, maxiter=200) -> torch.Tensor:
    if not depth:
        return laplace_jacobi_smoothen(f, bc, tol, maxiter)
    f = laplace_jacobi_smoothen(f, bc, tol, maxiter)
    f = interpolate(laplace_multigrid_f_cycle(restrict(f), bc.restrict(), depth-1, tol, maxiter))
    f = laplace_multigrid_v_cycle(f, bc, depth, tol, maxiter)
    return f

# W-Cycle
def laplace_multigrid_w_cycle(f: torch.Tensor, bc: BoundaryCondition, depth=4, tol=1e-5, maxiter=200) -> torch.Tensor:
    if not depth:
        return laplace_jacobi_smoothen(f, bc, tol, maxiter)
    f = laplace_jacobi_smoothen(f, bc, tol, maxiter)
    f = laplace_multigrid_w_cycle(restrict(f), bc.restrict(), depth-1, tol, maxiter)
    f = interpolate(laplace_multigrid_w_cycle(f, bc.restrict(), depth-1, tol, maxiter))
    f = laplace_jacobi_smoothen(f, bc, tol, maxiter)
    return f

# Full Multigrid V-Cycle
def laplace_full_multigrid_v_cycle(f: torch.Tensor, bc: BoundaryCondition, depth=4, tol=1e-5, maxiter=200) -> torch.Tensor:
    if not depth:
        return laplace_jacobi_smoothen(f, bc, tol, maxiter)
    f = interpolate(laplace_full_multigrid_v_cycle(restrict(f), bc.restrict(), depth-1, tol, maxiter))
    f = laplace_multigrid_v_cycle(f, bc, depth, tol, maxiter)
    return f