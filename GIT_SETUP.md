# Git Repository Setup Guide

This guide will help you publish the GSW PyTorch package to GitHub.

## Current Status

✅ Git repository initialized
✅ All files committed to `main` branch
✅ Release tag `v0.1.0` created
⏳ Remote repository needs to be set up

## Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right → "New repository"
3. Repository settings:
   - **Name**: `gsw-torch` (or your preferred name)
   - **Description**: "PyTorch implementation of the Gibbs SeaWater (GSW) Oceanographic Toolbox"
   - **Visibility**: Public (recommended) or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click "Create repository"

## Step 2: Add Remote and Push

After creating the repository, GitHub will show you commands. Use these commands:

```bash
cd /home/jrm22n/gsw2torch/implementation

git remote add origin https://github.com/0jrm/gsw-torch.git

# Or if using SSH:
# git remote add origin git@github.com:YOUR_USERNAME/gsw-torch.git

# Push main branch
git push -u origin main

# Push tags
git push origin v0.1.0
```

## Step 3: Create GitHub Release

1. Go to your repository on GitHub
2. Click "Releases" → "Draft a new release"
3. Select tag: `v0.1.0`
4. Title: `Version 0.1.0`
5. Description: Copy from `CHANGELOG.md` section for 0.1.0
6. Mark as "Latest release"
7. Click "Publish release"

## Step 4: Update pyproject.toml URLs (Optional)

After creating the repository, update the URLs in `pyproject.toml`:

```toml
[project.urls]
Homepage = "https://github.com/YOUR_USERNAME/gsw-torch"
Documentation = "https://github.com/YOUR_USERNAME/gsw-torch#readme"
Repository = "https://github.com/YOUR_USERNAME/gsw-torch"
Issues = "https://github.com/YOUR_USERNAME/gsw-torch/issues"
```

Then commit and push:
```bash
git add pyproject.toml
git commit -m "Update repository URLs"
git push
```

## Quick Commands Reference

```bash
# Check status
git status

# View commits
git log --oneline

# View tags
git tag -l

# Add remote (one-time setup)
git remote add origin https://github.com/YOUR_USERNAME/gsw-torch.git

# Push main branch
git push -u origin main

# Push all tags
git push origin --tags

# Push specific tag
git push origin v0.1.0

# View remotes
git remote -v

# Remove remote (if needed)
git remote remove origin
```

## Troubleshooting

### If remote already exists:
```bash
# Check current remote
git remote -v

# Update remote URL
git remote set-url origin https://github.com/YOUR_USERNAME/gsw-torch.git
```

### If you need to force push (use with caution):
```bash
git push -f origin main
```

### If authentication fails:
- Use GitHub Personal Access Token instead of password
- Or set up SSH keys: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

## Next Steps After Publishing

1. ✅ Repository is on GitHub
2. ✅ Release tag is published
3. ⏳ Set up GitHub Actions secrets (if publishing to PyPI)
4. ⏳ Update README badges with correct repository URLs
5. ⏳ Consider adding GitHub Topics: `oceanography`, `pytorch`, `thermodynamics`, `teos-10`

## Publishing to PyPI (Future)

When ready to publish to PyPI:

1. Create account on [PyPI](https://pypi.org) and [TestPyPI](https://test.pypi.org)
2. Generate API tokens
3. Add secrets to GitHub repository:
   - `PYPI_API_TOKEN` - for PyPI
   - `TEST_PYPI_API_TOKEN` - for TestPyPI
4. Update GitHub Actions workflow to publish on release tags

See `RELEASE.md` for detailed PyPI publishing instructions.
