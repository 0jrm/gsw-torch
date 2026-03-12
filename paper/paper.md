---
title: 'GSW-Torch: A Differentiable PyTorch Implementation of the TEOS-10 Gibbs SeaWater Toolbox'
tags:
  - Python
  - PyTorch
  - oceanography
  - TEOS-10
  - automatic differentiation
  - physics-informed machine learning
  - seawater thermodynamics
authors:
  - name: Jose R. Miranda
    orcid: 0000-0003-1133-4500
    corresponding: true
    affiliation: 1
  - name: Olmo Zavala-Romero
    orcid: 0000-0001-7088-3820
    affiliation: 1
affiliations:
  - name: Department of Earth, Ocean and Atmospheric Science, Florida State University, USA
    index: 1
date: 11 March 2026
bibliography: paper.bib
---

# Summary

The Gibbs SeaWater (GSW) Oceanographic Toolbox provides the standard
thermodynamic description of seawater under TEOS-10, the international
thermodynamic equation of seawater adopted by the Intergovernmental
Oceanographic Commission in 2009 [@IOC2010]. It is used throughout
physical oceanography for computing quantities such as in-situ density,
Conservative Temperature, Absolute Salinity, and dozens of derived
properties and their mutual conversions. Existing implementations in
MATLAB, C, Python, Julia, R, and Rust cover a wide range of computational
contexts, but none supports automatic differentiation (autograd).

`gsw-torch` is a pure PyTorch reimplementation of the GSW toolbox that
makes every function differentiable via PyTorch's autograd engine. The
library exposes 118 functions spanning conversions, density, energy,
freezing, ice thermodynamics, stability, geostrophy, interpolation, and
utility calculations. All implementations use `torch.float64` arithmetic
and are validated to be numerically equivalent to the C-backed reference
GSW-Python [@Firing2021] to within $10^{-8}$. Because the entire
computation graph is composed of PyTorch tensor operations, gradients
flow through any combination of GSW calculations automatically, enabling
their use in gradient-based optimization and physics-informed machine
learning (ML) workflows.

# Statement of Need

Modern ocean science increasingly relies on differentiable programming
and physics-informed ML. Physics-informed neural networks (PINNs)
[@Raissi2019], neural operators, and differentiable data assimilation
systems all require that physical equations be embedded inside a gradient
computation graph. In oceanography, seawater thermodynamics is a
necessary ingredient: density stratification, thermohaline properties,
and ocean state estimation all depend on the equations codified in
TEOS-10.

However, all current GSW implementations are built for forward evaluation
only. The Python package [@Firing2021] and the Julia, R, and Rust
[@Castelao2024] implementations are either thin wrappers around a C
library or non-differentiable native implementations. Although one could
in principle approximate gradients via finite differences, this is
numerically unstable for higher-order derivatives, does not scale to
large batch dimensions, and is incompatible with the end-to-end gradient
flow expected by frameworks like PyTorch.

`gsw-torch` fills this gap. It is the first implementation of TEOS-10 that
supports exact automatic differentiation, enabling researchers to:

- train physics-informed neural networks constrained by TEOS-10
  thermodynamics;
- compute analytical sensitivities (e.g., $\partial \rho / \partial S_A$,
  $\partial C_T / \partial \theta$) without finite-difference
  approximations;
- build differentiable ocean models for variational data assimilation;
- use GPU-accelerated seawater thermodynamics in large-scale ML pipelines.

The library targets the growing community working at the intersection of
deep learning and physical oceanography [@Reichstein2019; @Beucler2021;
@Karniadakis2021].

# State of the Field

The TEOS-10 toolbox has been ported to many languages. The canonical
Matlab implementation (GSW-m, @McDougall2011) remains the most complete
reference, but is proprietary. GSW-C [@Delahoyde2022] provides an
open-source C library that underlies the Python [@Firing2021], Julia
[@Barth2020], and R [@Kelley2022] packages, all of which are wrappers
around it. GSW-rs [@Castelao2024] is a recent pure Rust implementation
targeting embedded systems and microcontrollers. None of these
implementations supports differentiable computation.

`gsw-torch` is complementary, not competing, with these tools. For
forward-only evaluation, C-backed GSW-Python is substantially faster
than a pure-Python/PyTorch implementation. The trade-off is intentional:
`gsw-torch` sacrifices some raw throughput to gain the ability to propagate
gradients through the entire seawater equation of state. In batch ML
settings, where the same thermodynamic expressions are evaluated millions
of times on a GPU, this trade-off becomes favorable.

A summary of the landscape is given in Table 1.

| Implementation | Language | Differentiable | GPU | Open Source |
|---|---|---|---|---|
| GSW-m [@McDougall2011] | MATLAB | No | No | No |
| GSW-C [@Delahoyde2022] | C | No | No | Yes |
| GSW-Python [@Firing2021] | Python/C | No | No | Yes |
| GibbsSeaWater.jl [@Barth2020] | Julia | No | No | Yes |
| GSW-R [@Kelley2022] | R | No | No | Yes |
| GSW-rs [@Castelao2024] | Rust | No | No | Yes |
| **gsw-torch (this work)** | **Python/PyTorch** | **Yes** | **Yes** | **Yes** |

Table: Comparison of TEOS-10 GSW implementations.

# Software Design

## Architecture

`gsw-torch` mirrors the GSW-Python public API so that existing code can
be adapted by changing the import and replacing NumPy arrays with
`torch.Tensor`. The package is organised into public API modules
(`conversions`, `density`, `energy`, `freezing`, `ice`, `stability`,
`geostrophy`, `interpolation`, `utility`) that delegate to internal
`_core/` implementations. This separation allows the public interface to
remain stable while the numerical internals evolve.

## Pure PyTorch constraint

No NumPy is used inside the library itself; all arithmetic operates on
PyTorch tensors. This is the fundamental requirement for autograd
compatibility. The conversion utilities (`as_tensor`,
`match_args_return`) transparently accept NumPy arrays or Python scalars
at the boundary and convert them to tensors, so the interface remains
convenient without compromising differentiability.

## Autograd-safe implementation patterns

Maintaining autograd compatibility through complex thermodynamic
expressions requires several deliberate choices:

- **No `.item()` calls or Python branching on tensor values.** These
  break the computation graph. Conditionals are expressed with
  `torch.where`.
- **No in-place operations** that would corrupt gradient bookkeeping.
- **Preserved evaluation order.** The translation follows the reference
  GSW-Python source step-by-step to conserve floating-point rounding
  behaviour.
- **Fixed-iteration solvers.** Iterative solvers (e.g., for
  Conservative Temperature at freezing) use a predetermined number of
  Newton-Raphson steps rather than a convergence criterion, making
  the graph depth deterministic and finite.

## Numerical precision

All computations use `torch.float64` by default. Parity against
GSW-Python is verified at absolute tolerance $10^{-8}$ for the vast
majority of functions. A small number of functions have known precision
limitations: `enthalpy_second_derivatives` (`h_SA_SA` component)
accumulates errors from second-order autograd through square-root
operations ($\lesssim 4\times10^{-3}$ at 5000 dbar); `entropy_from_CT`
and the spiciness functions (`spiciness0/1/2`) exhibit errors of order
$10^{-7}$–$10^{-6}$ arising from polynomial approximations. These are
documented in `IMPLEMENTATION_STATUS.md` along with recommended
workarounds.

## Testing and continuous integration

Correctness is enforced through three layers of automated testing:

1. **Parity tests** compare every function against the reference
   GSW-Python output over representative oceanographic domains.
2. **Gradient checks** use `torch.autograd.gradcheck` to verify
   analytical gradients against numerical finite differences for all
   differentiable functions.
3. **Integration tests** exercise multi-step workflows such as
   stratification calculations and geostrophic velocity computation.

GitHub Actions CI runs the full test suite across Python 3.9, 3.10, and
3.11 on every commit, including linting with `ruff` and type checking
with `mypy`.

# Research Impact

`gsw-torch` enables a new category of ocean science applications.
Differentiable seawater thermodynamics is a prerequisite for embedding
the equation of state inside neural network training loops, computing
exact adjoint sensitivities for variational assimilation, or
constructing thermodynamically consistent generative models of ocean
state. While `gsw-torch` is slower than C-backed GSW for sequential
forward evaluation, GPU execution largely closes the gap for the
large-batch scenarios typical of ML training.

The package is published on PyPI as `gsw-torch` and is installable via
`pip install gsw-torch`. It is licensed under the BSD 3-Clause license,
consistent with the upstream TEOS-10 GSW licence. The source repository
is publicly available at `https://github.com/0jrm/gsw-torch`.

# Agentic AI Development

`gsw-torch` was developed using an agentic AI workflow: the translation
from GSW-Python to PyTorch was carried out autonomously by AI agents
within the Cursor IDE, operating under a strict set of human-defined
rules and guardrails. This methodology represents an emerging approach
to scientific software development and is disclosed here in accordance
with JOSS policy.

The agents were given the following non-negotiable constraints: (1) the
reference GSW-Python implementation was treated as a read-only oracle
that may not be imported in library code; (2) every translated function
must pass numerical parity tests at tolerance $10^{-8}$ before being
accepted; (3) no NumPy may be used inside the source code; (4) evaluation order
must be preserved to maintain rounding behaviour; (5) autograd
compatibility patterns (no `.item()`, no in-place ops, `torch.where` for
conditionals) must be respected throughout.

Under these constraints, the agents iteratively ported all 118 functions,
ran the test suite after each change, and self-corrected when tests
failed. Human review was applied to architectural decisions and to
any case where the agent reported a known limitation. The resulting
codebase passed comprehensive parity and gradient checks, demonstrating
that agentic AI can produce scientifically rigorous numerical code when
paired with well-specified validation criteria.

# Acknowledgements

The authors thank the TEOS-10 community and the developers of GSW-Python
[@Firing2021] for providing the reference implementation against which
`gsw-torch` is validated. Development was assisted by Cursor AI agents (model Composer 1). This work was supported by Florida State University.

# References
