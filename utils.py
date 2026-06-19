
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from dataclasses import dataclass
from typing import overload

def expand_to_nd(t: torch.Tensor, n: int) -> torch.Tensor:
    if n <= t.ndim: return t
    return t.reshape((1,)*(n-t.ndim) + t.shape)

# convolutions
def conv2d(g: torch.Tensor, h: torch.Tensor, stride=1) -> torch.Tensor:
    return F.conv2d(expand_to_nd(g, 3), expand_to_nd(h, 4), stride=stride)[0]

def convT2d(g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return F.conv_transpose2d(expand_to_nd(g, 3), expand_to_nd(h, 4), stride=2, padding=1)[0]

def conv1d(g: torch.Tensor, h: torch.Tensor, stride=1) -> torch.Tensor:
    return F.conv1d(expand_to_nd(g, 3), expand_to_nd(h, 3), stride=stride)[0]

def convT1d(g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return F.conv_transpose1d(expand_to_nd(g, 3), expand_to_nd(h, 4), stride=2, padding=1)[0]

# kernels
interp_kernel_2d = torch.tensor(((0.25, 0.5, 0.25), (0.5, 1., 0.5), (0.25, 0.5, 0.25)))
neighbor_kernel_2d = torch.tensor(((0., 0.25, 0.), (0.25, 0., 0.25), (0., 0.25, 0.)))
restrict_kernel_2d = torch.tensor(((.0625, .125, .0625), (.125, .25, .125), (.0625, .125, .0625)))
restrict_kernel_1d = torch.tensor((.25, .5, .25))

# padding
def reflection_pad_2d(f: torch.Tensor) -> torch.Tensor:
    return F.pad(f.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')[0, 0]

def reflection_pad_1d(f: torch.Tensor) -> torch.Tensor:
    return F.pad(f.unsqueeze(0).unsqueeze(0), (1, 1), mode='reflect')[0, 0]

def repl_pad_2d(f: torch.Tensor) -> torch.Tensor:
    return F.pad(f.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')[0, 0]

def repl_pad_1d(f: torch.Tensor) -> torch.Tensor:
    return F.pad(f.unsqueeze(0).unsqueeze(0), (1, 1), mode='replicate')[0, 0]

def zero_pad_2d(f: torch.Tensor) -> torch.Tensor:
    return F.pad(f, (1, 1, 1, 1))

def zero_pad_1d(f: torch.Tensor) -> torch.Tensor:
    return F.pad(f, (1, 1))

# other

def lerp(t1: torch.Tensor, t2: torch.Tensor, n: int = 16) -> torch.Tensor:
    assert t1.shape == t2.shape
    t = torch.linspace(0, 1, n).reshape([n] + [1]*t1.ndim)
    return (1-t)*t1 + t*t2

# up/downsampling
def even_pad(f: torch.Tensor, pad_h=True, pad_w=True) -> torch.Tensor:
    kernel = torch.ones(2 if pad_h else 3, 2 if pad_w else 3)
    kernel /= kernel.sum()
    return conv2d(repl_pad_2d(f), kernel)

def interpolate(f: torch.Tensor, og_shape=None) -> torch.Tensor:
    upconv = convT2d(f, interp_kernel_2d)
    if og_shape is None: return upconv
    og_h, og_w = og_shape
    if og_h%2 and og_w%2: return upconv
    return even_pad(upconv, not og_h%2, not og_w%2)

def restrict2d(f: torch.Tensor) -> torch.Tensor:
    return conv2d(repl_pad_2d(f), restrict_kernel_2d, stride=2)

def restrict2d_nopad(f: torch.Tensor) -> torch.Tensor:
    return conv2d(f, restrict_kernel_2d, stride=2)

def restrict1d(f: torch.Tensor) -> torch.Tensor:
    return conv1d(repl_pad_1d(f), restrict_kernel_1d, stride=2)

def restrict1d_nopad(f: torch.Tensor) -> torch.Tensor:
    return conv1d(f, restrict_kernel_1d, stride=2)

full_slice = slice(None, None, None)

@dataclass
class WeightedTensor:
    weight: torch.Tensor
    vals: torch.Tensor

    def apply(self, other: torch.Tensor) -> torch.Tensor:
        return (1-self.weight) * other + self.weight * self.vals

    def sum(self) -> torch.Tensor:
        return torch.sum(self.weight * self.vals)
    
    def numel(self) -> torch.Tensor:
        return torch.sum(self.weight)
    
    def __setitem__(self, accessor, value):
        self.weight[accessor] = 1
        self.vals[accessor] = value

    def __getitem__(self, key):
        return self.vals[key]
    
    def restrict(self) -> WeightedTensor:
        match self.vals.ndim:
            case 1:
                return WeightedTensor(restrict1d(self.weight), (restrict1d(self.vals * self.weight) / restrict1d(self.weight)).nan_to_num(0))
            case 2:
                return WeightedTensor(restrict2d(self.weight), (restrict2d(self.vals * self.weight) / restrict2d(self.weight)).nan_to_num(0))
            case _:
                raise NotImplementedError()
    
    def restrict_nopad(self) -> WeightedTensor:
        match self.vals.ndim:
            case 1:
                return WeightedTensor(restrict1d_nopad(self.weight), (restrict1d_nopad(self.vals * self.weight) / restrict1d_nopad(self.weight)).nan_to_num(0))
            case 2:
                return WeightedTensor(restrict2d_nopad(self.weight), (restrict2d_nopad(self.vals * self.weight) / restrict2d_nopad(self.weight)).nan_to_num(0))
            case _:
                raise NotImplementedError()
    
    def repl_pad(self) -> WeightedTensor:
        match self.vals.ndim:
            case 1:
                return WeightedTensor(repl_pad_1d(self.weight), repl_pad_1d(self.vals))
            case 2:
                return WeightedTensor(repl_pad_2d(self.weight), repl_pad_2d(self.vals))
            case _:
                raise NotImplementedError()

    @staticmethod
    def empty_w_size(*size) -> WeightedTensor:
        return WeightedTensor(torch.zeros(*size), torch.zeros(*size))

# boundary conditions
@dataclass
class DirichletBoundaryCondition:
    _inner: WeightedTensor
    _left: WeightedTensor
    _right: WeightedTensor
    _bottom: WeightedTensor
    _top: WeightedTensor

    def restrict(self) -> DirichletBoundaryCondition:
        return DirichletBoundaryCondition(
            self.full().restrict_nopad(),
            self.left.restrict(),
            self.right.restrict(),
            self.bottom.restrict(),
            self.top.restrict(),
        )
    
    @staticmethod
    def empty_w_size(height, width) -> DirichletBoundaryCondition:
        return DirichletBoundaryCondition(
            WeightedTensor.empty_w_size(height, width),
            WeightedTensor.empty_w_size(height),
            WeightedTensor.empty_w_size(height),
            WeightedTensor.empty_w_size(width),
            WeightedTensor.empty_w_size(width)
        )

    @staticmethod
    def empty_for_grid(grid: torch.Tensor) -> DirichletBoundaryCondition:
        return DirichletBoundaryCondition.empty_w_size(*grid.shape)
    
    def __setitem__(self, accessor, value):
        self._inner[accessor] = value

    def mean(self) -> torch.Tensor:
        sumel = (self._inner.sum() + self._left.sum() + self._right.sum() + self._bottom.sum() + self._top.sum())
        numel = (self._inner.numel() + self._left.numel() + self._right.numel() + self._bottom.numel() + self._top.numel())
        return sumel / numel
    
    def apply_inner(self, grid: torch.Tensor) -> torch.Tensor:
        return self._inner.apply(grid)
    
    def full(self) -> WeightedTensor:
        vals = zero_pad_2d(self._inner.vals)
        weight = zero_pad_2d(self._inner.weight)
        vals[0, 1:-1] = self._bottom.vals
        weight[0, 1:-1] = self._bottom.weight
        vals[-1, 1:-1] = self._top.vals
        weight[-1, 1:-1] = self._top.weight
        vals[1:-1, 0] = self._left.vals
        weight[1:-1, 0] = self._left.weight
        vals[1:-1, -1] = self._right.vals
        weight[1:-1, -1] = self._right.weight
        vals[0, 0] = 0.5*(vals[0, 1]+vals[1, 0])
        weight[0, 0] = 0.5*(weight[0, 1]+weight[1, 0])
        vals[0, -1] = 0.5*(vals[0, -2]+vals[1, -1])
        weight[0, -1] = 0.5*(weight[0, -2]+weight[1, -1])
        vals[-1, 0] = 0.5*(vals[-1, 1]+vals[-2, 0])
        weight[-1, 0] = 0.5*(weight[-1, 1]+weight[-2, 0])
        vals[-1, -1] = 0.5*(vals[-1, -2]+vals[-2, -1])
        weight[-1, -1] = 0.5*(weight[-1, -2]+weight[-2, -1])
        return WeightedTensor(weight, vals)
    
    def boundary_pad(self, grid: torch.Tensor) -> torch.Tensor:
        refl = reflection_pad_2d(grid)
        refl[0, 1:-1] = self._bottom.apply(refl[0, 1:-1])
        refl[-1, 1:-1] = self._top.apply(refl[-1, 1:-1])
        refl[1:-1, 0] = self._left.apply(refl[1:-1, 0])
        refl[1:-1, -1] = self._right.apply(refl[1:-1, -1])
        return refl

    def apply_full(self, grid: torch.Tensor) -> torch.Tensor:
        return self.full().apply(grid)
    
    @property
    def left(self) -> WeightedTensor:
        return self._left
    @left.setter
    def left(self, value):
        self._left[full_slice] = value
    
    @property
    def right(self) -> WeightedTensor:
        return self._right
    @right.setter
    def right(self, value):
        self._right[full_slice] = value
    
    @property
    def bottom(self) -> WeightedTensor:
        return self._bottom
    @bottom.setter
    def bottom(self, value):
        self._bottom[full_slice] = value
    
    @property
    def top(self) -> WeightedTensor:
        return self._top
    @top.setter
    def top(self, value):
        self._top[full_slice] = value
    
    def show(self, ax=None):
        if not ax:
            ax = plt
        full = self.full()
        ax.imshow(full.vals, origin='lower', alpha=full.weight)
    
    def smoothen(self, grid: torch.Tensor) -> torch.Tensor:
        smoothed = conv2d(self.boundary_pad(grid), neighbor_kernel_2d)
        return self.apply_inner(smoothed)
    
    def add_rectangle(self, width, height, value, center=None, cx=None, cy=None, left=None, right=None, top=None, bottom=None):
        assert center is not None or (cx is not None and cy is not None) or ((left is not None or right is not None) and (bottom is not None or top is not None))
        if center is not None:
            cx, cy = center
        if (cx and cy):
            left = cx - width // 2
            bottom = cy - height // 2
        if right:
            left = right - width
        if top:
            bottom = top - height
        
        self[bottom:bottom+height, left:left+width] = value

class CylinderBoundaryCondition:
    pass


@dataclass
class NeumannBoundaryCondition:
    left: torch.Tensor
    right: torch.Tensor
    bottom: torch.Tensor
    top: torch.Tensor

    def restrict(self) -> NeumannBoundaryCondition:
        return NeumannBoundaryCondition(self.left[::2], self.right[::2], self.bottom[::2], self.top[::2])
    
    @staticmethod
    def from_vector_field(v, coord_grid: torch.Tensor) -> NeumannBoundaryCondition:
        return NeumannBoundaryCondition(
            left   = -v(coord_grid[:, 0])[:, 0],
            right  = +v(coord_grid[:, -1])[:, 0],
            bottom = -v(coord_grid[0, :])[:, 1],
            top    = +v(coord_grid[-1, :])[:, 1],
        )

def neumann_nd(bc: NeumannBoundaryCondition) -> torch.Tensor:
    h, w = bc.left.numel(), bc.bottom.numel()
    nd = torch.zeros(h, w)
    nd[0, :] += bc.bottom
    nd[-1, :] += bc.top
    nd[:, 0] += bc.left
    nd[:, -1] += bc.right
    return nd

def laplace_smoothen(f: torch.Tensor) -> torch.Tensor:
    return conv2d(reflection_pad_2d(f), neighbor_kernel_2d)

def poisson_smoothen(u: torch.Tensor, f: torch.Tensor, bc: NeumannBoundaryCondition, h: float) -> torch.Tensor:
    return conv2d(repl_pad_2d(u), neighbor_kernel_2d) + (h / 4) * neumann_nd(bc) - (h**2) * f

def open_image(fname: str) -> torch.Tensor:
    img = Image.open(fname)
    return pil_to_tensor(img)

if __name__ == '__main__':
    test = torch.randn(3, 3)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(test, vmin=test.min(), vmax=test.max())
    ax2.imshow(reflection_pad_2d(test), vmin=test.min(), vmax=test.max())
    ax3.imshow(laplace_smoothen(test), vmin=test.min(), vmax=test.max())
    plt.show()