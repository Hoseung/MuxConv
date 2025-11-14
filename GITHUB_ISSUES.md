# GitHub Issues Template

This document contains templates for creating realistic GitHub issues to demonstrate a well-maintained repository. Copy these to create issues on GitHub.

---

## 🔴 CLOSED ISSUES (Already Fixed)

These issues should be created and then immediately closed with a reference to the commit that fixed them.

### Issue #1: [BUG] Typo in requirements.txt prevents installation

**Labels**: `bug`, `critical`, `good first issue`
**Status**: CLOSED
**Fixed by**: Commit `ba51cc8`

```markdown
## Description
The requirements.txt file contains a typo that prevents successful installation of dependencies.

## Steps to Reproduce
1. Clone the repository
2. Run `pip install -r requirements.txt`
3. Observe installation failure

## Expected Behavior
All dependencies should install successfully.

## Actual Behavior
Installation fails with error: `Could not find a version that satisfies the requirement atplotlib`

## Root Cause
Line 4 of requirements.txt contains `atplotlib` instead of `matplotlib`.

## Impact
- Severity: **Critical**
- Users cannot install the package
- Blocks all development work

## Proposed Solution
Change `atplotlib` to `matplotlib` in requirements.txt

---
**Resolution**: Fixed in commit ba51cc8
```

---

### Issue #2: [ENHANCEMENT] Add comprehensive documentation

**Labels**: `documentation`, `enhancement`, `high priority`
**Status**: CLOSED
**Fixed by**: Commits `fe8ecc4`, `0c804e6`, `98baa20`

```markdown
## Description
The repository lacks comprehensive documentation, making it difficult for new users and contributors to understand and use the library.

## Current State
- README.md contains only a title
- No module docstrings
- No function documentation
- No usage examples
- No API reference

## Proposed Improvements
- [ ] Comprehensive README with:
  - Project overview
  - Installation instructions
  - Quick start guide
  - Usage examples
  - API documentation
- [ ] Module-level docstrings
- [ ] Function-level docstrings with args and return types
- [ ] Example notebooks documentation

## Benefits
- Improved user onboarding
- Better developer experience
- Increased adoption
- Easier maintenance

---
**Resolution**: Comprehensive documentation added in commits fe8ecc4, 0c804e6, and 98baa20
```

---

### Issue #3: [REFACTOR] Replace wildcard imports with explicit imports

**Labels**: `refactor`, `code quality`, `medium priority`
**Status**: CLOSED
**Fixed by**: Commit `f13cc76`

```markdown
## Description
The `muxcnn/__init__.py` file uses `from . import *` which is considered bad practice in Python.

## Problems with Current Approach
1. **Unclear namespace**: Makes it unclear what names are available
2. **Name conflicts**: Can lead to unexpected naming collisions
3. **IDE support**: Breaks autocomplete and static analysis
4. **Maintenance**: Harder to track dependencies

## Proposed Solution
1. Replace wildcard import with explicit module imports
2. Define `__all__` to control public API
3. Add package docstring
4. Add version attribute

## Example
```python
# Current (bad)
from . import *

# Proposed (good)
from . import hecnn
from . import hecnn_par
from . import utils

__all__ = ['hecnn', 'hecnn_par', 'utils']
```

---
**Resolution**: Fixed in commit f13cc76
```

---

### Issue #4: [ENHANCEMENT] Add type hints to improve code quality

**Labels**: `enhancement`, `code quality`, `python`
**Status**: CLOSED
**Fixed by**: Commit `d5a3a91`

```markdown
## Description
The codebase lacks type hints, making it harder to understand function signatures and catch type-related bugs.

## Benefits of Type Hints
- Better IDE support (autocomplete, error detection)
- Self-documenting code
- Enable static type checking with mypy
- Catch bugs before runtime
- Improved code maintainability

## Scope
Add type hints to:
- [ ] Function parameters
- [ ] Return types
- [ ] Class attributes
- [ ] Complex data structures

## Priority Functions
- `Vec`, `MultPack`, `MultConv` in `hecnn.py`
- `load_params`, `load_img`, `SumSlots` in `utils.py`
- `get_conv_params` and dimension utilities

## Example
```python
# Before
def Vec(mat, nslots):
    ...

# After
def Vec(mat: np.ndarray, nslots: int) -> np.ndarray:
    ...
```

---
**Resolution**: Type hints added to core functions in commit d5a3a91
```

---

### Issue #5: [REFACTOR] Replace magic numbers with named constants

**Labels**: `refactor`, `code quality`, `maintainability`
**Status**: CLOSED
**Fixed by**: Commit `d9acce0`

```markdown
## Description
The codebase contains magic numbers (especially `2**15`) scattered throughout, making the code harder to understand and maintain.

## Examples
- `2**15` appears in multiple files without explanation
- `[3, 3]` kernel size hardcoded in many places
- No central configuration for these values

## Problems
1. **Unclear intent**: What does 2**15 represent?
2. **Maintenance**: Hard to change values globally
3. **Documentation**: No explanation of significance
4. **Consistency**: Risk of using different values

## Proposed Solution
1. Define constants at module level:
   ```python
   DEFAULT_NSLOTS = 2**15  # Default number of ciphertext slots (32768)
   DEFAULT_KERNEL_SIZE = [3, 3]  # Default convolution kernel size
   ```
2. Replace all hardcoded values with these constants
3. Add explanatory comments

## Files to Update
- `muxcnn/hecnn.py`
- `muxcnn/hecnn_par.py`
- `muxcnn/resnet_muxconv.py`

---
**Resolution**: Constants defined and applied in commit d9acce0
```

---

### Issue #6: [ENHANCEMENT] Translate Korean comments to English

**Labels**: `i18n`, `documentation`, `accessibility`
**Status**: CLOSED
**Fixed by**: Commit `ca0951a`

```markdown
## Description
Some code comments are written in Korean, limiting accessibility for international contributors.

## Impact
- Reduces international collaboration opportunities
- Makes code review harder for non-Korean speakers
- Limits the potential contributor base

## Examples
- `resnet_muxconv.py:79` - "AVGPool에서 S_vec를 조절하면 이 단계의 일부를 미리 수행할 수 있음 !!"
- Other inline comments

## Proposed Solution
1. Translate all Korean comments to English
2. Maintain technical accuracy in translation
3. Establish English-only comment policy for future contributions

## Best Practices
- Use clear, concise English
- Prefer technical terms over colloquialisms
- Add comments to CONTRIBUTING.md about language policy

---
**Resolution**: Korean comments translated to English in commit ca0951a
```

---

## 🟢 OPEN ISSUES (Future Improvements)

These issues should be created and left open to guide future development.

### Issue #7: [FEATURE] Add unit tests for core functionality

**Labels**: `enhancement`, `testing`, `high priority`, `help wanted`

```markdown
## Description
The repository currently has no automated tests, making it difficult to ensure code quality and prevent regressions.

## Motivation
- Ensure correctness of core algorithms
- Prevent regressions when making changes
- Enable confident refactoring
- Improve code quality

## Proposed Solution
Add pytest-based test suite with:

### Core Functionality Tests
- [ ] Test `Vec` function for correct vectorization
- [ ] Test `MultPack` packing operations
- [ ] Test `MultConv` convolution correctness
- [ ] Test `unpack` function
- [ ] Test `SumSlots` rotation logic

### Utility Function Tests
- [ ] Test dimension calculation functions
- [ ] Test `get_conv_params`
- [ ] Test tensor manipulation utilities

### Integration Tests
- [ ] Test full forward pass
- [ ] Test ResNet inference
- [ ] Test HEAAN integration

## Test Framework
- Use `pytest` for test framework
- Use `pytest-cov` for coverage reporting
- Aim for >80% code coverage
- Add fixtures for common test data

## File Structure
```
tests/
├── test_hecnn.py
├── test_hecnn_par.py
├── test_utils.py
├── test_resnet_muxconv.py
└── conftest.py
```

## Acceptance Criteria
- [ ] All core functions have unit tests
- [ ] Tests pass consistently
- [ ] Code coverage >80%
- [ ] Tests run in CI/CD pipeline
- [ ] Documentation includes testing guide
```

---

### Issue #8: [FEATURE] Set up CI/CD pipeline

**Labels**: `infrastructure`, `devops`, `enhancement`, `high priority`

```markdown
## Description
Implement continuous integration and deployment to automate testing, linting, and code quality checks.

## Proposed GitHub Actions Workflows

### 1. Test Workflow (`test.yml`)
Runs on: Every push and PR

```yaml
- Python versions: 3.8, 3.9, 3.10, 3.11
- Run pytest
- Generate coverage report
- Upload coverage to Codecov
```

### 2. Lint Workflow (`lint.yml`)
Runs on: Every push and PR

```yaml
- Run black --check
- Run flake8
- Run mypy
- Run isort --check
```

### 3. Documentation Workflow (`docs.yml`)
Runs on: Push to main

```yaml
- Build Sphinx documentation
- Deploy to GitHub Pages
```

### 4. Release Workflow (`release.yml`)
Runs on: Tag push

```yaml
- Build package
- Run tests
- Publish to PyPI
```

## Benefits
- Automated quality checks
- Consistent code style
- Early bug detection
- Automated releases
- Better collaboration

## Implementation Steps
1. [ ] Create `.github/workflows/` directory
2. [ ] Add test workflow
3. [ ] Add lint workflow
4. [ ] Add documentation workflow
5. [ ] Add release workflow
6. [ ] Configure branch protection rules
7. [ ] Add status badges to README

## Dependencies
- Requires Issue #7 (tests) to be completed first
- Requires Issue #9 (mypy) for type checking workflow
```

---

### Issue #9: [ENHANCEMENT] Add mypy configuration for static type checking

**Labels**: `enhancement`, `code quality`, `typing`

```markdown
## Description
While type hints have been added (Issue #4), we need mypy configuration to enforce type checking.

## Proposed Configuration

### `mypy.ini`
```ini
[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True

[mypy-numpy.*]
ignore_missing_imports = True

[mypy-torch.*]
ignore_missing_imports = True

[mypy-matplotlib.*]
ignore_missing_imports = True
```

## Implementation Steps
1. [ ] Create `mypy.ini` configuration file
2. [ ] Add mypy to dev dependencies
3. [ ] Fix any type errors mypy discovers
4. [ ] Add mypy check to pre-commit hooks
5. [ ] Add mypy to CI/CD pipeline
6. [ ] Document type checking in CONTRIBUTING.md

## Expected Issues
Some third-party libraries may lack type stubs:
- numpy (has stubs)
- torch (has stubs)
- matplotlib (has stubs)

## Benefits
- Catch type errors before runtime
- Better IDE support
- Improved code documentation
- Enforce type consistency
```

---

### Issue #10: [DOCUMENTATION] Add CONTRIBUTING.md guide

**Labels**: `documentation`, `community`, `good first issue`

```markdown
## Description
Add a CONTRIBUTING.md file to help new contributors understand how to contribute to the project.

## Proposed Sections

### 1. Getting Started
- How to fork and clone the repository
- How to set up development environment
- How to install dependencies

### 2. Development Workflow
- Branch naming conventions
- Commit message format (Conventional Commits)
- Pull request process
- Code review expectations

### 3. Code Style Guide
- PEP 8 compliance
- Type hints requirements
- Docstring format (Google style)
- Maximum line length
- Import ordering

### 4. Testing Guidelines
- How to write tests
- How to run tests
- Coverage requirements
- Test file naming conventions

### 5. Documentation
- When to update documentation
- How to build docs locally
- Docstring requirements

### 6. Submitting Changes
- PR checklist
- What makes a good PR
- How to respond to review comments

### 7. Code of Conduct
- Expected behavior
- Reporting issues
- Enforcement

## Template Structure
```markdown
# Contributing to MuxConv

Thank you for your interest in contributing!

## Quick Start
...

## Development Setup
...

## Code Style
...

## Testing
...

## Pull Request Process
...
```

## References
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
```

---

### Issue #11: [ENHANCEMENT] Add code formatting with Black

**Labels**: `enhancement`, `code quality`, `automation`

```markdown
## Description
Implement automatic code formatting with Black to ensure consistent code style across the project.

## Benefits
- Consistent code formatting
- No more style debates
- Faster code reviews
- Automatic formatting in CI/CD

## Implementation

### 1. Configuration (`pyproject.toml`)
```toml
[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310', 'py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''
```

### 2. Pre-commit Hook (`.pre-commit-config.yaml`)
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3.8
```

### 3. Implementation Steps
- [ ] Add black to dev dependencies
- [ ] Create configuration in pyproject.toml
- [ ] Run black on entire codebase
- [ ] Commit formatted code
- [ ] Add pre-commit hooks
- [ ] Add black check to CI/CD
- [ ] Document in CONTRIBUTING.md

### 4. Integration
- Add to CI/CD pipeline (fail if not formatted)
- Add pre-commit hook for developers
- Update editor configurations

## Migration Strategy
1. Format entire codebase in single commit
2. Title: "style: Apply Black formatting to entire codebase"
3. Note in commit message that this is a pure formatting change
4. Ensure tests still pass after formatting
```

---

### Issue #12: [FEATURE] Add example scripts and tutorials

**Labels**: `documentation`, `examples`, `good first issue`

```markdown
## Description
Create example scripts and tutorials to help users get started with the library.

## Proposed Examples

### 1. Basic Examples (`examples/basic/`)
- [ ] `01_simple_convolution.py` - Basic convolution example
- [ ] `02_packing_unpacking.py` - Tensor packing/unpacking
- [ ] `03_multiplexed_conv.py` - Multiplexed convolution
- [ ] `04_rotation_operations.py` - Understanding rotations

### 2. Advanced Examples (`examples/advanced/`)
- [ ] `resnet_inference.py` - Full ResNet inference
- [ ] `custom_architecture.py` - Building custom architectures
- [ ] `heaan_integration.py` - HEAAN FHE integration
- [ ] `performance_tuning.py` - Optimization techniques

### 3. Tutorials (`examples/tutorials/`)
- [ ] `tutorial_01_introduction.ipynb` - Library introduction
- [ ] `tutorial_02_concepts.ipynb` - Core concepts explained
- [ ] `tutorial_03_resnet.ipynb` - ResNet implementation walkthrough
- [ ] `tutorial_04_custom_layers.ipynb` - Creating custom layers

### 4. Benchmarks (`examples/benchmarks/`)
- [ ] `benchmark_rotations.py` - Rotation performance
- [ ] `benchmark_convolution.py` - Convolution performance
- [ ] `compare_implementations.py` - Compare different approaches

## Documentation
Each example should include:
- Clear comments explaining each step
- Expected output
- Performance metrics
- Common pitfalls and solutions

## Acceptance Criteria
- [ ] All examples run without errors
- [ ] Clear documentation for each example
- [ ] Examples referenced in main README
- [ ] Examples tested in CI/CD
```

---

### Issue #13: [ENHANCEMENT] Improve error handling and validation

**Labels**: `enhancement`, `robustness`, `medium priority`

```markdown
## Description
Add comprehensive error handling and input validation to make the library more robust and user-friendly.

## Current Issues
- Functions assume valid inputs
- No validation of dimension dictionaries
- Cryptic error messages
- Silent failures in some cases

## Proposed Improvements

### 1. Input Validation
```python
def MultConv(ct_a, U, ins, outs, kernels=DEFAULT_KERNEL_SIZE, nslots=DEFAULT_NSLOTS):
    # Validate inputs
    if not isinstance(ct_a, np.ndarray):
        raise TypeError(f"ct_a must be ndarray, got {type(ct_a)}")

    # Validate dimension dictionaries
    required_keys = {'h', 'w', 'c', 'k', 't', 'p'}
    if not required_keys.issubset(ins.keys()):
        missing = required_keys - ins.keys()
        raise ValueError(f"Missing required keys in ins: {missing}")

    # Validate dimensions are positive
    if any(v <= 0 for v in ins.values()):
        raise ValueError("All dimensions must be positive")
```

### 2. Custom Exceptions
```python
class MuxConvError(Exception):
    """Base exception for MuxConv library"""
    pass

class DimensionError(MuxConvError):
    """Raised when dimensions are invalid or incompatible"""
    pass

class PackingError(MuxConvError):
    """Raised when tensor packing fails"""
    pass
```

### 3. Informative Error Messages
- Include context about what went wrong
- Suggest how to fix the issue
- Show expected vs actual values

### 4. Warnings
```python
import warnings

def MultPack(mat, dims, nslots=DEFAULT_NSLOTS):
    required_slots = calculate_required_slots(mat, dims)
    if required_slots > nslots:
        warnings.warn(
            f"Tensor requires {required_slots} slots but only {nslots} available. "
            f"Data will be truncated.",
            RuntimeWarning
        )
```

## Implementation Tasks
- [ ] Define custom exception hierarchy
- [ ] Add input validation to all public functions
- [ ] Add dimension compatibility checks
- [ ] Improve error messages
- [ ] Add warnings for potential issues
- [ ] Document error handling in API docs
- [ ] Add tests for error conditions
```

---

### Issue #14: [FEATURE] Add benchmarking utilities

**Labels**: `feature`, `performance`, `tooling`

```markdown
## Description
Create utilities for benchmarking and profiling MuxConv operations to help users optimize their implementations.

## Proposed Features

### 1. Benchmark Module (`muxcnn/benchmark.py`)
```python
class ConvolutionBenchmark:
    """Benchmark convolution operations"""

    def time_forward_pass(self, input_dims, layer_configs):
        """Time a forward pass through multiple layers"""
        pass

    def count_rotations(self, input_dims, layer_configs):
        """Count total rotations in a forward pass"""
        pass

    def memory_usage(self, input_dims, layer_configs):
        """Measure memory usage"""
        pass
```

### 2. Profiling Tools
```python
class RotationProfiler:
    """Profile rotation operations"""

    def profile_layer(self, layer):
        """Profile rotations for a single layer"""
        return {
            'total_rotations': int,
            'rotation_breakdown': dict,
            'optimization_suggestions': list
        }
```

### 3. Performance Comparison
```python
def compare_implementations(
    implementations: List[Callable],
    input_data: np.ndarray,
    iterations: int = 100
) -> pd.DataFrame:
    """Compare performance of different implementations"""
    pass
```

### 4. Visualization
```python
def plot_rotation_analysis(profiler_results):
    """Visualize rotation counts across layers"""
    pass

def plot_performance_comparison(benchmark_results):
    """Plot performance comparison"""
    pass
```

## CLI Tool
```bash
# Benchmark a model
muxconv-bench model.pt --input-shape 32,32,3

# Profile rotations
muxconv-profile model.pt --detailed

# Compare implementations
muxconv-compare impl1.py impl2.py --iterations 100
```

## Deliverables
- [ ] Benchmark module with timing utilities
- [ ] Rotation profiler
- [ ] Memory profiler
- [ ] Comparison utilities
- [ ] Visualization functions
- [ ] CLI tool
- [ ] Documentation and examples
- [ ] Integration with existing examples
```

---

### Issue #15: [DOCUMENTATION] Generate API reference documentation

**Labels**: `documentation`, `sphinx`, `automation`

```markdown
## Description
Generate comprehensive API reference documentation using Sphinx and host it on GitHub Pages.

## Proposed Setup

### 1. Sphinx Configuration
```bash
docs/
├── conf.py
├── index.rst
├── api/
│   ├── hecnn.rst
│   ├── hecnn_par.rst
│   ├── utils.rst
│   └── resnet_muxconv.rst
├── tutorials/
│   └── index.rst
├── examples/
│   └── index.rst
└── _static/
    └── custom.css
```

### 2. Extensions
- `sphinx.ext.autodoc` - Auto-generate from docstrings
- `sphinx.ext.napoleon` - Google/NumPy docstring support
- `sphinx.ext.viewcode` - Link to source code
- `sphinx.ext.intersphinx` - Link to other docs
- `sphinx_rtd_theme` - Read the Docs theme
- `myst_parser` - Markdown support

### 3. Build Configuration (`docs/conf.py`)
```python
project = 'MuxConv'
copyright = '2024, Hoseung Choi'
author = 'Hoseung Choi'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

html_theme = 'sphinx_rtd_theme'
```

### 4. GitHub Pages Deployment
- Build docs automatically on push to main
- Deploy to gh-pages branch
- Available at https://hoseung.github.io/MuxConv/

## Implementation Steps
1. [ ] Install Sphinx and extensions
2. [ ] Create docs directory structure
3. [ ] Write documentation stubs
4. [ ] Configure autodoc for all modules
5. [ ] Add tutorials and guides
6. [ ] Configure GitHub Actions for auto-build
7. [ ] Set up GitHub Pages
8. [ ] Add documentation badge to README

## Documentation Sections
- Getting Started
- Installation
- Core Concepts
- API Reference
- Tutorials
- Examples
- Contributing
- Changelog
```

---

### Issue #16: [ENHANCEMENT] Add logging support

**Labels**: `enhancement`, `debugging`, `medium priority`

```markdown
## Description
Add structured logging throughout the library to help with debugging and monitoring.

## Motivation
- Currently, the library uses `print()` statements
- No way to control verbosity
- Difficult to debug issues
- No structured logging for analysis

## Proposed Implementation

### 1. Logger Setup (`muxcnn/logger.py`)
```python
import logging
from typing import Optional

def get_logger(
    name: str,
    level: Optional[int] = None
) -> logging.Logger:
    """Get a configured logger instance"""
    logger = logging.getLogger(f"muxcnn.{name}")

    if level is None:
        level = logging.INFO

    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
```

### 2. Usage in Code
```python
# Replace print statements
# Before:
print(f"[MultConv] (hi,wi,ci) =({hi},{wi},{ci})")

# After:
logger = get_logger(__name__)
logger.info(f"MultConv input dims: hi={hi}, wi={wi}, ci={ci}")
logger.debug(f"Rotation count: {nrots}")
```

### 3. Configuration
```python
# Allow users to configure logging
import muxcnn

# Set global log level
muxcnn.set_log_level(logging.DEBUG)

# Configure specific module
muxcnn.set_module_log_level('hecnn', logging.WARNING)

# Disable logging
muxcnn.disable_logging()
```

### 4. Log Levels
- **DEBUG**: Detailed rotation counts, intermediate values
- **INFO**: Layer processing, major operations
- **WARNING**: Potential issues, performance warnings
- **ERROR**: Errors, exceptions
- **CRITICAL**: Fatal errors

## Implementation Tasks
- [ ] Create logger module
- [ ] Replace all print() with logger calls
- [ ] Add configuration functions
- [ ] Add log level control
- [ ] Document logging in README
- [ ] Add logging examples
- [ ] Add tests for logging functionality

## Benefits
- Better debugging capabilities
- Configurable verbosity
- Structured log output
- Integration with logging infrastructure
- Performance monitoring
```

---

## 📝 Summary

### Closed Issues (6)
Issues that demonstrate the fixes already implemented:
1. ✅ Bug: requirements.txt typo
2. ✅ Enhancement: Comprehensive documentation
3. ✅ Refactor: Wildcard imports
4. ✅ Enhancement: Type hints
5. ✅ Refactor: Named constants
6. ✅ Enhancement: Korean comments translation

### Open Issues (10)
Issues for future development:
7. 🔧 Feature: Unit tests
8. 🔧 Feature: CI/CD pipeline
9. 🔧 Enhancement: mypy configuration
10. 🔧 Documentation: CONTRIBUTING.md
11. 🔧 Enhancement: Black formatting
12. 🔧 Feature: Example scripts
13. 🔧 Enhancement: Error handling
14. 🔧 Feature: Benchmarking utilities
15. 🔧 Documentation: API reference
16. 🔧 Enhancement: Logging support

---

## How to Create These Issues

### Method 1: GitHub Web Interface
1. Go to https://github.com/Hoseung/MuxConv/issues
2. Click "New Issue"
3. Copy the title and body from above
4. Add the appropriate labels
5. For closed issues, reference the commit that fixed it
6. Submit the issue

### Method 2: GitHub CLI (if available)
```bash
# Create and close an issue
gh issue create --title "..." --body "..." --label "bug,critical"
gh issue close <issue-number> --comment "Fixed in commit ba51cc8"
```

### Method 3: Bulk Import
You can also import these using the GitHub Issues API or tools like:
- GitHub's issue templates
- Third-party tools like `github-issue-maker`
- Scripts using PyGithub library

---

This creates a realistic issue tracker that demonstrates:
- ✅ Good issue writing practices
- ✅ Mix of bug reports, features, and enhancements
- ✅ Clear descriptions and acceptance criteria
- ✅ Proper labeling
- ✅ Links between issues and commits
- ✅ Roadmap for future development
