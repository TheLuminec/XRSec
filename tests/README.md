# Tests

## Structure
The test suite is organized as:

- `tests/unit/` for fast isolated unit tests.
- `tests/integration/` for cross-module or data-dependent tests.
- `tests/fixtures/` for reusable fixture assets.

## Running fast unit tests
Run only fast unit tests (excluding slow tests):

```bash
pytest -m "unit and not slow"
```

## Running integration tests
Run integration tests only:

```bash
pytest -m integration
```

## Local prerequisites
Some integration tests may require local artifacts that are not committed to the repository:

- Dataset files under `datasets/` (download/populate as needed for your environment).
- Model checkpoints such as `model/trained_model.pth` or files under `model/saved/`.

If these prerequisites are missing, integration tests should be skipped or are expected to fail locally.
