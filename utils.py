"""
utils.py
--------
Shared utility functions used across the project:
- Printing formatted banners / separators.
- Reading environment variables with friendly error messages.
- Validating GitHub URLs.
- Displaying sources from retrieved documents.
"""

import os
import re
from typing import List


# ── Visual helpers ────────────────────────────────────────────────────────────

def print_banner() -> None:
    """Print a startup banner for the CLI application."""
    banner = r"""
╔══════════════════════════════════════════════════════╗
║        🔍  AI GitHub Repo Explainer  🔍              ║
║  Ask anything about any public GitHub repository!    ║
╚══════════════════════════════════════════════════════╝
"""
    print(banner)


def print_separator(char: str = "─", width: int = 60) -> None:
    """Print a visual separator line."""
    print(char * width)


def print_sources(sources: List[str]) -> None:
    """
    Display the list of source files that contributed to an answer.

    Args:
        sources: List of relative file paths.
    """
    if not sources:
        return
    print("\n📎  Sources used:")
    for src in sorted(sources):
        print(f"    • {src}")


# ── Environment / Config ──────────────────────────────────────────────────────

def get_openai_key() -> str:
    """
    Retrieve the OpenAI API key from the environment.

    Returns:
        The API key string, or an empty string if not set.
    """
    return os.environ.get("OPENAI_API_KEY", "")


def require_openai_key() -> str:
    """
    Retrieve the OpenAI API key and raise a clear error if it is missing.

    Returns:
        The API key string.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
    """
    key = get_openai_key()
    if not key:
        raise EnvironmentError(
            "❌  OPENAI_API_KEY environment variable is not set.\n"
            "    Set it with:  set OPENAI_API_KEY=sk-...   (Windows)\n"
            "                  export OPENAI_API_KEY=sk-...  (Linux/macOS)"
        )
    return key


# ── Validation ────────────────────────────────────────────────────────────────

# Matches e.g. https://github.com/owner/repo  (with optional .git suffix)
_GITHUB_URL_PATTERN = re.compile(
    r"^https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(\.git)?/?$"
)


def validate_github_url(url: str) -> bool:
    """
    Basic validation that a string looks like a GitHub repository URL.

    Args:
        url: URL string to validate.

    Returns:
        True if valid, False otherwise.
    """
    return bool(_GITHUB_URL_PATTERN.match(url.strip()))


def normalize_github_url(url: str) -> str:
    """
    Strip trailing slashes and optional .git suffix from a GitHub URL
    so GitPython and the UI both get a canonical form.

    Args:
        url: Raw URL entered by the user.

    Returns:
        Cleaned URL string.
    """
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url


# ── User Input Helpers ────────────────────────────────────────────────────────

def prompt_yes_no(question: str, default: bool = True) -> bool:
    """
    Ask the user a yes/no question and return the boolean result.

    Args:
        question: The question string (without y/n hint, that's added automatically).
        default:  What to return when the user presses Enter without input.

    Returns:
        True for yes, False for no.
    """
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{question} {hint}: ").strip().lower()
        if answer in ("", "y", "yes"):
            return True if (answer == "" and default) or answer in ("y", "yes") else False
        if answer in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")
