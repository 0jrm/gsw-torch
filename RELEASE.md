# Release Process

This document describes the process for releasing new versions of GSW PyTorch.

## Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality in a backwards-compatible manner
- **PATCH** version for backwards-compatible bug fixes

## Pre-Release Checklist

Before creating a release, ensure:

- [ ] All tests pass (`pytest`)
- [ ] Code quality checks pass (`ruff check`, `mypy gsw_torch`)
- [ ] Documentation is up to date
- [ ] `CHANGELOG.md` is updated with release notes
- [ ] `README.md` reflects current status
- [ ] Version number updated in:
  - `pyproject.toml`
  - `gsw_torch/_version.py`
- [ ] `LICENSE` file is present
- [ ] `MANIFEST.in` includes all necessary files
- [ ] Package builds successfully (`python -m build`)
- [ ] Package installs successfully (`pip install dist/*.whl`)
- [ ] Import works (`python -c "import gsw_torch; print(gsw_torch.__version__)"`)

## Release Steps

### 1. Update Version

Update version in two places:

**`pyproject.toml`:**
```toml
[project]
version = "0.1.0"  # Update to new version
```

**`gsw_torch/_version.py`:**
```python
__version__ = "0.1.0"  # Update to new version
```

### 2. Update CHANGELOG

Add a new section to `CHANGELOG.md`:

```markdown
## [0.1.0] - 2025-01-27

### Added
- List of new features

### Changed
- List of changes

### Fixed
- List of bug fixes

### Known Limitations
- List any known issues
```

### 3. Build Package

```bash
# Install build tools
pip install build

# Build source distribution and wheel
python -m build

# Verify build
ls dist/
# Should see: gsw-torch-0.1.0.tar.gz and gsw_torch-0.1.0-py3-none-any.whl
```

### 4. Test Installation

```bash
# Create clean virtual environment
python -m venv test_env
source test_env/bin/activate  # Linux/Mac
# or
test_env\Scripts\activate  # Windows

# Install from wheel
pip install dist/gsw_torch-0.1.0-py3-none-any.whl

# Test import
python -c "import gsw_torch; print(gsw_torch.__version__)"
python -c "import gsw_torch; help(gsw_torch.CT_from_t)"
```

### 5. Create Git Tag

```bash
# Commit all changes
git add .
git commit -m "Release version 0.1.0"

# Create annotated tag
git tag -a v0.1.0 -m "Release version 0.1.0"

# Push commits and tags
git push origin main
git push origin v0.1.0
```

### 6. Create GitHub Release

1. Go to GitHub repository
2. Click "Releases" → "Draft a new release"
3. Select tag: `v0.1.0`
4. Title: `Version 0.1.0`
5. Description: Copy from `CHANGELOG.md`
6. Upload `dist/gsw-torch-0.1.0.tar.gz` and `dist/gsw_torch-0.1.0-py3-none-any.whl`
7. Mark as "Latest release" if appropriate
8. Publish release

### 7. Publish to PyPI (Future)

**Note**: PyPI publishing should be done carefully and only after thorough testing.

```bash
# Install twine
pip install twine

# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ gsw-torch

# If successful, upload to PyPI
twine upload dist/*
```

**PyPI Credentials**: Store in `~/.pypirc` or use environment variables.

## Post-Release

After release:

1. Update `IMPLEMENTATION_STATUS.md` if needed
2. Announce release (if applicable)
3. Monitor for issues
4. Plan next release

## Release Types

### Major Release (X.0.0)

- Breaking API changes
- Major new features
- Significant refactoring
- Requires migration guide

### Minor Release (0.X.0)

- New functions added
- New features
- Backwards compatible
- Update `IMPLEMENTATION_STATUS.md`

### Patch Release (0.0.X)

- Bug fixes
- Documentation updates
- Performance improvements
- No API changes

## Hotfix Process

For critical bugs:

1. Create hotfix branch from main
2. Fix the issue
3. Add test to prevent regression
4. Follow release process with patch version bump
5. Merge hotfix back to main

## Version 0.1.0 Release Notes

### Initial Release

- 118+ functions implemented
- Full PyTorch implementation with autograd support
- GPU acceleration support
- Numerical parity with reference GSW
- Comprehensive test suite
- Complete documentation

### Known Limitations

See `IMPLEMENTATION_STATUS.md` for detailed list of known limitations, including:
- Precision issues in some derivative functions
- Autograd limitations for edge cases (SA=0)
- Polynomial fitting precision in spiciness functions

## Future Releases

### Planned for 0.2.0

- Additional functions from reference GSW
- Performance optimizations
- Enhanced documentation
- API improvements based on user feedback

### Planned for 1.0.0

- Complete function coverage
- Stable API
- Production-ready status
- Full documentation
