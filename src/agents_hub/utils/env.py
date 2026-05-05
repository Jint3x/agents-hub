"""Environment variable loader with .env support."""

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env(env_path: Path | str | None = None) -> None:
    """Load environment variables from a .env file.

    Args:
        env_path: Explicit path to the .env file. If None, searches upward
            from the current working directory for a .env file.
    """
    if env_path is not None:
        load_dotenv(dotenv_path=str(env_path), override=True)
        return

    cwd = Path.cwd()
    for directory in (cwd, *cwd.parents):
        dotenv_file = directory / ".env"
        if dotenv_file.exists():
            load_dotenv(dotenv_path=str(dotenv_file), override=True)
            return


def get_required_env(key: str) -> str:
    """Return the value of a required environment variable.

    Args:
        key: Name of the environment variable.

    Raises:
        RuntimeError: If the variable is not set or empty.
    """
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value
