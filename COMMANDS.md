# Commands

Every command someone working on `middleware-llmobs` is likely to need — dev loop, packaging,
and the release flow. Copy/paste-able, no surprises.

---

## Dev environment

```bash
# Create + activate a venv (Python ≥ 3.10, < 3.15)
python -m venv .venv
source .venv/bin/activate

# Editable install with test extras
pip install -e ".[test]"

# Dev tooling not declared in pyproject (the workflows install these on CI)
pip install ruff mypy
```

## The check loop (mirrors CI exactly)

Run these before pushing — the workflows run the same three commands and will fail the build
otherwise:

```bash
ruff check src tests
ruff format --check src tests
mypy src/middleware/llmobs
pytest -q
```

Quick fixers:

```bash
ruff check --fix src tests        # auto-fix lint
ruff format src tests             # auto-format
```

## Running examples

```bash
cd examples
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export OTEL_EXPORTER_OTLP_ENDPOINT="https://uid.middleware.io:443"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=<your-middleware-key>"
export OPENAI_API_KEY="<your-openai-key>"

python 01_basic_tracing/app.py
python 02_toxicity_llm_judge/app.py
```

## Build (local sanity check)

```bash
pip install hatch
hatch version                     # what version are we on?
hatch build -t sdist -t wheel     # produces dist/*.tar.gz + dist/*.whl
ls dist/
```

To clean the build outputs:

```bash
rm -rf dist build *.egg-info
```

## Bumping the version

The version is **static** in `pyproject.toml` and the CI build verifies it matches the tag.
Edit it before tagging:

```bash
# pyproject.toml -> [project] version = "0.2.0rc1"
# Commit the bump:
git add pyproject.toml
git commit -m "chore: bump version to 0.2.0rc1"
```

> If you use `hatch version <X>` instead of editing by hand, hatch will rewrite the file for
> you — but `version` must be declared as a string literal under `[project]` (which it is).

---

## Releasing

The repo ships two workflows; the tag glob decides which one runs.

| Workflow | Trigger | Behavior |
|---|---|---|
| `.github/workflows/rc-publish.yml` | `v*rc*` tag push | test → build → publish to PyPI with `skip-existing` |
| `.github/workflows/release.yml`    | `v*` tag push that doesn't contain `rc` | test → build → publish to PyPI (no skip) |

### Release candidate (`v*rc*`)

```bash
# 1. Bump version in pyproject.toml to e.g. 0.2.0rc1 and commit.
git commit -am "chore: bump version to 0.2.0rc1"

# 2. Tag it. Tag MUST match pyproject's version (the CI build verifies this).
git tag -a v0.2.0rc1 -m "Release candidate 0.2.0rc1"

# 3. Push the tag (rc workflow fires on this push, not on the commit).
git push origin main
git push origin v0.2.0rc1
```

Watch the run at `https://github.com/middleware-labs/middleware-llmobs/actions`. If everything
passes, `pip install middleware-llmobs==0.2.0rc1` (or `--pre`) works within a minute.

### Final release (`v*`, no `rc`)

```bash
# 1. Bump version in pyproject.toml to the final number, e.g. 0.2.0, and commit.
git commit -am "chore: bump version to 0.2.0"

# 2. Tag + push.
git tag -a v0.2.0 -m "Release 0.2.0"
git push origin main
git push origin v0.2.0
```

A duplicate final-release tag will **fail** the publish step (intentional — `skip-existing` is
not set on the release workflow).

### Inspect / re-run / fix

```bash
# List remote tags
git ls-remote --tags origin

# Delete a tag locally + remotely (e.g. you tagged the wrong commit)
git tag -d v0.2.0rc1
git push --delete origin v0.2.0rc1

# Re-run a failed CI without re-tagging — use the "Re-run jobs" button in Actions UI,
# or trigger manually:
gh workflow run rc-publish.yml --ref main
gh workflow run release.yml    --ref main
```

### Required repo secrets

Configure once at **Settings → Secrets and variables → Actions**:

| Secret | Purpose |
|---|---|
| `PYPI_TOKEN` | PyPI API token scoped to the `middleware-llmobs` project. Used as the password with `user: __token__` (the PyPI standard for token auth). |
| `PYPI_USERNAME` | Reserved — workflows document the alternate username/password auth path but don't use it by default. |
| `PYPI_PASSWORD` | Same as above. |

To create the PyPI token: log into PyPI → Account settings → API tokens → **Add API token** →
Scope = "Project: middleware-llmobs".

---

## Git remote

Already wired up by `git init`:

```bash
git remote -v
# origin  https://github.com/middleware-labs/middleware-llmobs.git (fetch)
# origin  https://github.com/middleware-labs/middleware-llmobs.git (push)

# First push (after the first commit lands):
git push -u origin main
```
