# Contributing to BatchGen

Thank you for considering contributing to BatchGen!
We welcome contributions of all kinds from the community.
Whether you're introducing new features, enhancing the infrastructure, fixing bugs, or writing documentation, we appreciate your enthusiasm and value your efforts.

To help make your contributions as smooth as possible, we've put together this guide with helpful tips and best practices for contributing to the project.

## Table of Contents

- [Contributing to BatchGen](#contributing-to-batchgen)
  - [Table of Contents](#table-of-contents)
  - [How to Contribute](#how-to-contribute)
    - [Merge Policy](#merge-policy)
    - [Roadmap and Tasks](#roadmap-and-tasks)
    - [Development Environment](#development-environment)
    - [Commit Message Guidelines](#commit-message-guidelines)
      - [Commit Message Structure](#commit-message-structure)


## How to Contribute

- Check the [issue tracker](https://github.com/batchgen-project/batchgen/issues) for open issues, or open a new one to discuss your idea.
- Follow the [Fork-and-Pull-Request](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow when opening your pull requests.
- Ensure your code follows our style guidelines and passes all tests (see [Development Environment](#development-environment)).
- Submit a pull request with a clear description of your changes.
  - The pull request title should follow the [Commit Message Guidelines](#commit-message-guidelines).
  - The description should follow the [Pull Request Template](https://github.com/batchgen-project/batchgen/blob/main/.github/PULL_REQUEST_TEMPLATE.md).
  - Make sure to mention any related issues.

Before your pull request can be merged, it must pass the formatting, linting, and testing checks (see [Development Environment](#development-environment)).

### Merge Policy

To keep the `main` branch coherent and reviewed, only the project owner
presses the **Merge** button on pull requests. Contributors with `Write`
access — including members of the `batchgen-core` team — should:

- Open pull requests targeting `main`.
- Push commits to feature branches and PR branches as needed.
- Review pull requests, leave comments, formally Approve / Request changes.
- **Not press the Merge button** on any PR, including their own. Wait for
  the owner to merge after approval.

The owner is the only person whose merge lands on `main`. Pull requests
must have at least one approving review (from a Code Owner where
applicable, per `.github/CODEOWNERS`) and a green CI status before the
owner merges. Direct pushes to `main` are reserved for the owner only.

This policy is currently enforced socially. When the repository becomes
public it will be enforced by GitHub branch protection.

If you fix a bug:
- Add a relevant unit test when possible. These can be found in the `test` directory.
If you make an improvement:
- Update any affected example console scripts in the `examples` directory and documentation in the `docs` directory.
- Update unit tests when relevant.
If you add a feature:
- Include unit tests in the `test` directory.
- Add a demo script in the `examples` directory.

### Roadmap and Tasks

For beginners, we recommend starting with issues labeled `good first issue` or `help wanted` in the [issue tracker](https://github.com/batchgen-project/batchgen/issues).
Feel free to discuss any ideas before getting started!

### Development Environment

Ensure your development environment is set up with the following tools:

- Format your code with pre-commit hooks:
```bash
pip install -r requirements-lint.txt

# add lint hooks to git commit
pre-commit install --install-hooks
```

This will automatically format your code before committing. However, you can also run the following commands manually:
```bash
# format code
pre-commit run -a
```

- (Recommended) Sign off your commits:
```bash
git commit -s -m "feat: add new feature"
```

### Commit Message Guidelines

We follow the commit format rule based on the [Angular Commit Format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#-commit-message-format). This format improves readability and helps generate changelogs automatically.

#### Commit Message Structure

Each commit message should consist of a **header** and a **body**:

```
<type>: <summary>
<BLANK LINE>
<body>(optional)
<BLANK LINE>
```
- **Type**: Choose from `build`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `test`, `chore`.
- **Summary**: A brief description of the change.
- **Body**: Mandatory for all commits except those of type "docs". Must be at least 20 characters long.


Examples:

```
feat: add logging in sllm worker
```

```
docs: add new example for serving vision model

Vision mode: xxx
Implemented xxx in `xxx.py`
```

For more details, read the [Angular Commit Format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#-commit-message-format).
