# AGENTS

## Code Style
- Target Python 3.7 or later.
- Use 4 spaces for indentation and avoid tabs.
- Prefer `snake_case` for functions and variables and `CapWords` for classes.
- Keep imports grouped: standard library, third-party, then local modules.
- Include type hints and concise docstrings for public functions and classes.

## Testing
- Run `pytest -q` and ensure it passes before committing.

## Documentation
- Write comments and docstrings in English.
- Update `README.md` or other docs when behavior or APIs change.

## Dependencies
- Keep external dependencies minimal.
- If a new dependency is required, update both `pyproject.toml` and `requirements.txt`.
