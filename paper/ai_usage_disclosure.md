# AI Usage Disclosure

**Submission:** GSW-Torch: A Differentiable PyTorch Implementation of the TEOS-10 Gibbs SeaWater Toolbox  
**Journal:** Journal of Open Source Software

---

## Tools and Models Used

| Tool | Version | Where Used |
|---|---|---|
| Cursor IDE | composer 1 | Software code generation, refactoring, test writing, documentation drafting |
| Claude (Anthropic) | claude-sonnet-4-6 (via Cursor) | Paper text assistance |

---

## Nature and Scope of Assistance

### Software (library code in `gsw_torch/`)

The translation of GSW-Python functions into PyTorch was carried out in large part by AI agents operating within the Cursor IDE in an **agentic mode**: the agents were given access to the reference GSW-Python source code and a set of human-authored rules (stored in `.cursor/rules/`), and autonomously translated functions, ran the test suite, interpreted failures, and iterated until parity and gradient tests passed.

Specific tasks performed by the AI agents:

- **Code generation:** Translating each GSW-Python function from NumPy/C-backed implementations into pure PyTorch tensor operations across all nine modules (conversions, density, energy, freezing, ice, stability, geostrophy, interpolation, utility).
- **Refactoring:** Replacing NumPy idioms (`np.where`, `np.sqrt`, masked arrays) with autograd-compatible PyTorch equivalents (`torch.where`, `torch.sqrt`, `torch.clamp`).
- **Test scaffolding:** Generating parity test stubs, gradient check tests, and integration test workflows.
- **Documentation:** Drafting module-level and function-level docstrings.
- **Paper text:** Drafting sections of `paper.md` (this manuscript), subsequently reviewed and edited by the human authors.

### Documentation

AI assistance was used to draft `README.md`, `IMPLEMENTATION_STATUS.md`, `QUICKSTART.md`, and `CONTRIBUTING.md`. All documentation was reviewed and edited by the human authors.

### Paper

AI assistance (Claude via Cursor) was used to draft the initial text of all sections of `paper.md` and `paper.bib`. The human authors reviewed, substantially edited, and validated all content, including the accuracy of citations, the technical descriptions, and the framing of the contribution.

---

## Confirmation of Human Review

The human authors (Jose R. Miranda, Olmo Zavala-Romero) confirm the following:

1. **All AI-generated library code was validated** against the reference GSW-Python oracle using automated parity tests (tolerance $10^{-8}$) and gradient checks (`torch.autograd.gradcheck`) before acceptance. No function was merged into the codebase without passing these tests.

2. **Core design decisions were made by human authors**, including: the choice of PyTorch as the target framework, the requirement for pure-tensor implementations with no NumPy in library code, the `float64` default precision policy, the public API design mirroring GSW-Python, and the autograd-compatibility constraints (no `.item()`, no in-place ops, `torch.where` for conditionals).

3. **The human-authored rule set** (`.cursor/rules/`) governing agent behaviour was written entirely by the human authors and defines the scientific and engineering constraints that all AI-generated code must satisfy.

4. **All AI-generated paper text was reviewed, edited, and validated** by the human authors for scientific accuracy, completeness, and conformance with JOSS requirements.

5. **Known limitations** of the AI-generated implementation (documented in `IMPLEMENTATION_STATUS.md`) were identified through human review of test failures and are honestly disclosed in the paper.
