## A Multigrid library for 2D Laplace & Poisson equations

### 1. Getting started with Laplace solvers

The central defining property of a 2D Laplace equation is its set of **Boundary conditions**. This library supports rectangular domains with *Dirichlet boundary conditions* (=prescribed values) at the four edges of the domain, as well as in the interior. Start by initializing an empty Dirichlet boundary condition, either with a given size:

```python
bc = DirichletBoundaryCondition.empty_w_size(257, 257)
```

or for an existing rectangular grid (e.g. one created using ```torch.meshgrid```):

```python
bc = DirichletBoundaryCondition.empty_for_grid(grid)
```

Then, you can assign values to the boundary condition using standard PyTorch indexing semantics. Indexing ```bc``` directly accesses the interior, while ```left```, ```right```, ```top```, and ```bottom``` access the edges.

Here is a simple example:
```python
bc = DirichletBoundaryCondition.empty_w_size(256, 256)

bc.bottom[:128] = torch.linspace(13, 5, 128)
bc.bottom[128:192] = 5
bc.bottom[192:] = torch.linspace(5, 13, 64)

bc.top = 21

bc.left[:100] = torch.linspace(13, 40, 100)
bc.left[100:150] = 40
bc.left[150:] = torch.linspace(40, 21, 106)

bc.right[:150] = torch.linspace(13, 40, 150)
bc.right[150:200] = 40
bc.right[200:] = torch.linspace(40, 21, 56)

bc.add_rectangle(50, 50, 40, left=0, bottom=100)
bc.add_rectangle(50, 50, 40, right=-1, top=200)
```

The corresponding Laplace equation can then be solved with a multigrid solver of choice. This library includes V-Cycle, W-Cycle, F-Cycle, and Full Multigrid schemes.

```python
f_smooth = laplace_multigrid_half_v_cycle(torch.full((256, 256), 22.), bc, depth=4, tol=1e-4, maxiter=200, jacobi_damping_factor=2/3)
```

![Two heaters Laplace equation](/examples/two_heaters.png)