import torch
import torch.nn as nn
from dataclasses import dataclass
from utils import interpolate, restrict2d, laplace_smoothen, poisson_smoothen
from utils import DirichletBoundaryCondition, NeumannBoundaryCondition, neumann_nd
import scipy.sparse.linalg as spla
import numpy as np



def laplace_jacobi_step(f: torch.Tensor, bc: DirichletBoundaryCondition, damping_factor=1.0):
    """
    Apply a single Jacobi smoothing step to a given vector in the setting of a Laplace equation given boundary conditions.

    Parameters
    ----------
        f : torch.Tensor
            The function vector to smoothen
        bc : DirichletBoundaryCondition
            The boundary condition of the problem to solve
        damping_factor : float, optional
            The jacobi damping factor do be used (default: 1.0 / no damping)
    """
    if damping_factor == 1.0:
        smoothed =  bc.smoothen(f)
    else:
        smoothed = (1-damping_factor) * f + damping_factor * bc.smoothen(f)

    return smoothed


def laplace_jacobi_smoothen(f: torch.Tensor, bc: DirichletBoundaryCondition, damping_factor=1.0, tol=1e-3, maxiter=200, mae_log: list[float] | None = None) -> torch.Tensor:
    """
    Apply Jacobi smoothing step to a given vector to solve a Laplace equation with given boundary conditions, until a given residual threshold is reached or an iteration limit is hit

    Parameters
    ----------
        f : torch.Tensor
            The function vector to smoothen
        bc : DirichletBoundaryCondition
            The boundary condition of the problem to solve
        damping_factor : float, optional
            The jacobi damping factor do be used (default: 1.0 / no damping)
        tol : float, optional
            The residual threshold to use as termination condition (default: 0.001)
        maxiter : int, optional 
            The maximum number of iterations to perform (default: 200)
        mae_log: list[float], optional
            A list to log residual values
    """
    for i in range(maxiter):
        f_smooth = laplace_jacobi_step(f, bc, damping_factor)
        residual_mae = torch.mean(torch.abs(f - f_smooth)).item()
        if mae_log is not None:
            mae_log.append(residual_mae)
        if residual_mae <= tol:
            print(f'Smoothed for {i+1} iteration(s)')
            return f_smooth
        f = f_smooth
    print(f'Smoothed for {maxiter} iterations')
    return f


def poisson_jacobi_smoothen(u: torch.Tensor, f: torch.Tensor, bc: NeumannBoundaryCondition, h: float, tol=1e-3, maxiter=200) -> torch.Tensor:
    """
    Apply Jacobi smoothing step to a given vector to solve a Poisson equation with given boundary conditions, until a given residual threshold is reached or an iteration limit is hit

    Parameters
    ----------
        u : torch.Tensor
            The function vector to smoothen
        f : torch.Tensor
            The target vector of the Poisson equation
        bc : NeumannBoundaryConditions
            The boundary condition of the problem to solve
        h : float
            The grid spacing
        damping_factor : float, optional
            The jacobi damping factor do be used (default: 1.0 / no damping)
        tol : float, optional
            The residual threshold to use as termination condition (default: 0.001)
        maxiter : int, optional 
            The maximum number of iterations to perform (default: 200)
    """
    for _ in range(maxiter):
        u_smooth = poisson_smoothen(u, f, bc, h)
        if torch.mean(torch.abs(u - u_smooth)) <= tol:
            return u_smooth
        u = u_smooth
    return u

def poisson_exact_solve(f: torch.Tensor, bc: NeumannBoundaryCondition, h: float, u: torch.Tensor | None = None) -> torch.Tensor:
    """
    Solve a poisson equation exactly on a coarse grid.

    Parameters
    ----------
        u : torch.Tensor
            The function vector to smoothen
        f : torch.Tensor
            The target vector of the Poisson equation
        bc : NeumannBoundaryConditions
            The boundary condition of the problem to solve
        h : float
            The grid spacing
    """
    L = spla.LaplacianNd(f.shape, boundary_conditions='neumann').tosparse()
    b = (h**2) * f - h * neumann_nd(bc)
    solution, *_ = spla.lsqr(L, b.flatten(), x0=u.flatten()) if u is not None else spla.lsqr(L, b.flatten())
    return torch.from_numpy(solution).reshape_as(f).float()
    

### Laplace Solvers ###

# V-Cycle
def laplace_multigrid_v_cycle(f: torch.Tensor, bc: DirichletBoundaryCondition, depth=4, jacobi_damping_factor=1.0, tol=1e-5, maxiter=200, mae_log: list[float] | None = None) -> torch.Tensor:
    """
    Perform a V-Cycle iteration to solve a Laplace equation with Dirichlet boundary conditions.

    Parameters
    ----------
        f : torch.Tensor
            The initial guess for the function vector
        bc : DirichletBoundaryCondition
            The boundary condition of the problem to solve
        depth: int, optional
            The number of coarsening steps to apply (default: 4)
        damping_factor : float, optional
            The jacobi damping factor do be used (default: 1.0 / no damping)
        tol : float, optional
            The residual threshold to use as termination condition (default: 0.001)
        maxiter : int, optional 
            The maximum number of iterations to perform per smoothing epoch (default: 200)
        mae_log: list[float], optional
            A list to log residual values
    """
    if not depth:
        return laplace_jacobi_smoothen(f, bc, jacobi_damping_factor, tol, maxiter, mae_log)
    f = laplace_jacobi_smoothen(f, bc, jacobi_damping_factor, tol, maxiter, mae_log)
    f = interpolate(laplace_multigrid_v_cycle(restrict2d(f), bc.restrict(), depth-1, jacobi_damping_factor, tol, maxiter, mae_log), og_shape=f.shape)
    f = laplace_jacobi_smoothen(f, bc, jacobi_damping_factor, tol, maxiter, mae_log)
    return f

def laplace_multigrid_half_v_cycle(f: torch.Tensor, bc: DirichletBoundaryCondition, depth=4, jacobi_damping_factor=1.0, tol=1e-5, maxiter=200, mae_log: list[float] | None = None) -> torch.Tensor:
    """
    Perform a Half V-Cycle iteration to solve a Laplace equation with Dirichlet boundary conditions.

    Parameters
    ----------
        f : torch.Tensor
            The initial guess for the function vector
        bc : DirichletBoundaryCondition
            The boundary condition of the problem to solve
        depth: int, optional
            The number of coarsening steps to apply (default: 4)
        damping_factor : float, optional
            The jacobi damping factor do be used (default: 1.0 / no damping)
        tol : float, optional
            The residual threshold to use as termination condition (default: 0.001)
        maxiter : int, optional 
            The maximum number of iterations to perform per smoothing epoch (default: 200)
        mae_log: list[float], optional
            A list to log residual values
    """
    if not depth:
        return laplace_jacobi_smoothen(f, bc, jacobi_damping_factor, tol, maxiter, mae_log)
    f = interpolate(laplace_multigrid_half_v_cycle(restrict2d(f), bc.restrict(), depth-1, jacobi_damping_factor, tol, maxiter, mae_log), og_shape=f.shape)
    f = laplace_jacobi_smoothen(f, bc, jacobi_damping_factor, tol, maxiter, mae_log)
    return f

# F-Cycle
def laplace_multigrid_f_cycle(f: torch.Tensor, bc: DirichletBoundaryCondition, depth=4, jacobi_damping_factor=1.0, tol=1e-5, maxiter=200, mae_log: list[float] | None = None) -> torch.Tensor:
    """
    Perform a F-Cycle iteration to solve a Laplace equation with Dirichlet boundary conditions.

    Parameters
    ----------
        f : torch.Tensor
            The initial guess for the function vector
        bc : DirichletBoundaryCondition
            The boundary condition of the problem to solve
        depth: int, optional
            The number of coarsening steps to apply (default: 4)
        damping_factor : float, optional
            The jacobi damping factor do be used (default: 1.0 / no damping)
        tol : float, optional
            The residual threshold to use as termination condition (default: 0.001)
        maxiter : int, optional 
            The maximum number of iterations to perform per smoothing epoch (default: 200)
        mae_log: list[float], optional
            A list to log residual values
    """
    if not depth:
        return laplace_jacobi_smoothen(f, bc, jacobi_damping_factor, tol, maxiter, mae_log)
    f = laplace_jacobi_smoothen(f, bc, jacobi_damping_factor, tol, maxiter, mae_log)
    f = interpolate(laplace_multigrid_f_cycle(restrict2d(f), bc.restrict(), depth-1, jacobi_damping_factor, tol, maxiter, mae_log), og_shape=f.shape)
    f = laplace_multigrid_v_cycle(f, bc, depth, jacobi_damping_factor, tol, maxiter, mae_log)
    return f

# W-Cycle
def laplace_multigrid_w_cycle(f: torch.Tensor, bc: DirichletBoundaryCondition, depth=4, jacobi_damping_factor=1.0, tol=1e-5, maxiter=200, mae_log: list[float] | None = None) -> torch.Tensor:
    """
    Perform a W-Cycle iteration to solve a Laplace equation with Dirichlet boundary conditions.

    Parameters
    ----------
        f : torch.Tensor
            The initial guess for the function vector
        bc : DirichletBoundaryCondition
            The boundary condition of the problem to solve
        depth: int, optional
            The number of coarsening steps to apply (default: 4)
        damping_factor : float, optional
            The jacobi damping factor do be used (default: 1.0 / no damping)
        tol : float, optional
            The residual threshold to use as termination condition (default: 0.001)
        maxiter : int, optional 
            The maximum number of iterations to perform per smoothing epoch (default: 200)
        mae_log: list[float], optional
            A list to log residual values
    """
    if not depth:
        return laplace_jacobi_smoothen(f, bc, jacobi_damping_factor, tol, maxiter, mae_log)
    og_shape = f.shape
    f = laplace_jacobi_smoothen(f, bc, jacobi_damping_factor, tol, maxiter, mae_log)
    f = laplace_multigrid_w_cycle(restrict2d(f), bc.restrict(), depth-1, jacobi_damping_factor, tol, maxiter, mae_log)
    f = interpolate(laplace_multigrid_w_cycle(f, bc.restrict(), depth-1, jacobi_damping_factor, tol, maxiter, mae_log), og_shape=og_shape)
    f = laplace_jacobi_smoothen(f, bc, jacobi_damping_factor, tol, maxiter, mae_log)
    return f

# Full Multigrid V-Cycle
def laplace_full_multigrid_v_cycle(f: torch.Tensor, bc: DirichletBoundaryCondition, depth=4, jacobi_damping_factor=1.0, tol=1e-5, maxiter=200, mae_log: list[float] | None = None) -> torch.Tensor:
    """
    Perform a Full Multigrid V-Cycle iteration to solve a Laplace equation with Dirichlet boundary conditions.

    Parameters
    ----------
        f : torch.Tensor
            The initial guess for the function vector
        bc : DirichletBoundaryCondition
            The boundary condition of the problem to solve
        depth: int, optional
            The number of coarsening steps to apply (default: 4)
        damping_factor : float, optional
            The jacobi damping factor do be used (default: 1.0 / no damping)
        tol : float, optional
            The residual threshold to use as termination condition (default: 0.001)
        maxiter : int, optional 
            The maximum number of iterations to perform per smoothing epoch (default: 200)
        mae_log: list[float], optional
            A list to log residual values
    """
    if not depth:
        return laplace_jacobi_smoothen(f, bc, jacobi_damping_factor, tol, maxiter, mae_log)
    f = interpolate(laplace_full_multigrid_v_cycle(restrict2d(f), bc.restrict(), depth-1, jacobi_damping_factor, tol, maxiter, mae_log), og_shape=f.shape)
    f = laplace_multigrid_v_cycle(f, bc, depth, jacobi_damping_factor, tol, maxiter, mae_log)
    return f


### Poisson solvers ###

# # V-Cycle
# def poisson_multigrid_v_cycle(u: torch.Tensor, f: torch.Tensor, h: float, depth=4, tol=1e-5, maxiter=200) -> torch.Tensor:
#     if not depth:
#         return poisson_jacobi_smoothen(u, f, h, tol, maxiter)
#     f = poisson_jacobi_smoothen(u, f, h, tol, maxiter)
#     f = interpolate(poisson_multigrid_v_cycle(restrict(u), restrict(f), 2*h, depth-1, tol, maxiter))
#     f = poisson_jacobi_smoothen(u, f, h, tol, maxiter)
#     return f

# # Full Multigrid V-Cycle
# def poisson_full_multigrid_v_cycle(u: torch.Tensor, f: torch.Tensor, h: float, depth=4, tol=1e-5, maxiter=200) -> torch.Tensor:
#     if not depth:
#         return poisson_jacobi_smoothen(u, f, h, tol, maxiter)
#     f = interpolate(poisson_full_multigrid_v_cycle(restrict(u), restrict(f), 2*h, depth-1, tol, maxiter))
#     f = poisson_multigrid_v_cycle(u, f, h, depth, tol, maxiter)
#     return f

def poisson_multigrid(f: torch.Tensor, bc: NeumannBoundaryCondition, h: float, depth: int = 4, u: torch.Tensor | None = None, tol=1e-5, maxiter=200) -> torch.Tensor:
    """
    Apply a Half V-Cyle iteration to solve a Poisson equation with Neumann boundary conditions.

    Parameters
    ----------
        u : torch.Tensor
            The function vector to smoothen
        f : torch.Tensor
            The target vector of the Poisson equation
        bc : NeumannBoundaryConditions
            The boundary condition of the problem to solve
        h : float
            The grid spacing
        depth : int, optional
            The number of coarsening steps to apply (default: 4)
        tol : float, optional
            The residual threshold to use as termination condition (default: 0.001)
        maxiter : int, optional 
            The maximum number of iterations to perform (default: 200)
    """
    if not depth:
        return poisson_exact_solve(f, bc, h, u)
    u = interpolate(poisson_multigrid(restrict2d(f), bc.restrict(), 2*h, depth-1, restrict2d(u) if u is not None else None))
    u = poisson_jacobi_smoothen(u, f, bc, h, tol, maxiter)
    return u

def integrate_vf_on_grid(v, grid: torch.Tensor, multigrid_depth=4, tol=1e-5, maxiter=200) -> torch.Tensor:
    """
    Approximate the Helmholtz potential (up to a constant offset) of a given vector field.

    Parameters
    ----------
    v : Callable
        The vector field to integrate
    grid: torch.Tensor
        The grid on which to compute the Helmholtz potential
    multigrid_depth : int, optional
        The number of coarsening steps to apply (default: 4)
    tol : float, optional
        The residual threshold to use as termination condition (default: 0.001)
    maxiter : int, optional 
        The maximum number of iterations to perform (default: 200)
    """
    h1 = grid[0, 1, 0] - grid[0, 0, 0]
    h2 = grid[1, 0, 1] - grid[0, 0, 1]
    assert h1 == h2, f'Grid must be uniform! Horizontal grid spacing: {h1} does not match vertical grid spacing: {h2}.'
    div_v = torch.einsum(
        'x whx->wh',
        torch.autograd.functional.jacobian(lambda x: v(x).sum((0, 1)), grid)
    )
    bcond = NeumannBoundaryCondition.from_vector_field(v, grid)
    u = poisson_multigrid(div_v, bcond, h1.item(), depth=multigrid_depth, tol=tol, maxiter=maxiter)
    return u