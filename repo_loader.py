"""
repo_loader.py
--------------
Responsible for:
1. Cloning a GitHub repository to a local temp directory using GitPython.
2. Walking the cloned repo and reading all supported code/doc files.
3. Filtering out binary files, huge files, and unwanted directories.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

import git  # GitPython

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# File extensions we want to process
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",  # Python & JavaScript / TypeScript
    ".cpp", ".c", ".h", ".hpp",           # C / C++
    ".java",                               # Java
    ".go",                                 # Go
    ".rs",                                 # Rust
    ".md", ".txt", ".rst",                # Documentation
    ".yaml", ".yml", ".toml", ".json",    # Config / metadata
    ".html", ".css",                       # Web
    ".sh", ".bat",                         # Shell scripts
}

# Directories to skip entirely – they add noise and slow processing
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    ".env", "dist", "build", ".idea", ".vscode", ".mypy_cache",
    "site-packages", ".pytest_cache", "eggs", ".eggs",
}

# Maximum file size to process (bytes) – skip files larger than this
MAX_FILE_SIZE_BYTES = 200 * 1024  # 200 KB


def clone_repository(github_url: str) -> str:
    """
    Clone a public GitHub repository to a temporary local directory.

    Args:
        github_url: Full HTTPS URL of the GitHub repo
                    e.g. 'https://github.com/owner/repo'

    Returns:
        Absolute path to the cloned repo on disk.

    Raises:
        ValueError: If the URL looks invalid.
        git.GitCommandError: If cloning fails (e.g. private repo, bad URL).
    """
    if not github_url.startswith("https://") and not github_url.startswith("http://"):
        raise ValueError(
            f"Invalid GitHub URL: '{github_url}'. Must start with https:// or http://"
        )

    # Create a unique temporary directory that persists until we delete it
    clone_dir = tempfile.mkdtemp(prefix="repo_explainer_")
    print(f"\n📥  Cloning: {github_url}")
    print(f"📂  Destination: {clone_dir}")

    try:
        git.Repo.clone_from(github_url, clone_dir, depth=1)  # shallow clone = faster
        print("✅  Clone successful!\n")
    except git.GitCommandError as e:
        shutil.rmtree(clone_dir, ignore_errors=True)  # clean up on failure
        raise RuntimeError(
            f"Failed to clone '{github_url}'.\n"
            f"Make sure the URL is correct and the repo is public.\n"
            f"Git error: {e}"
        ) from e

    return clone_dir


def load_files_from_repo(repo_path: str) -> List[Tuple[str, str]]:
    """
    Walk a cloned repository directory and return (relative_path, content) pairs
    for every supported file within size limits.

    Args:
        repo_path: Absolute path to the cloned repository directory.

    Returns:
        List of (relative_file_path, file_content) tuples.
    """
    documents: List[Tuple[str, str]] = []
    repo_root = Path(repo_path)

    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Prune directories we don't want to descend into (modify in-place)
        dirnames[:] = [
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
        ]

        for filename in filenames:
            file_path = Path(dirpath) / filename
            suffix = file_path.suffix.lower()

            # Only process supported extensions
            if suffix not in SUPPORTED_EXTENSIONS:
                continue

            # Skip files that are too large
            try:
                file_size = file_path.stat().st_size
                if file_size > MAX_FILE_SIZE_BYTES:
                    print(f"⚠️  Skipping large file ({file_size // 1024} KB): {file_path.name}")
                    continue
            except OSError:
                continue  # Can't stat – skip

            # Read the file content
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if content.strip():  # skip empty files
                    relative_path = str(file_path.relative_to(repo_root))
                    documents.append((relative_path, content))
            except Exception as exc:
                print(f"⚠️  Could not read {file_path.name}: {exc}")
                continue

    print(f"📄  Loaded {len(documents)} file(s) from the repository.\n")
    return documents


def cleanup_repo(repo_path: str) -> None:
    """
    Remove the temporary cloned repository directory from disk.

    Args:
        repo_path: Path returned by clone_repository().
    """
    try:
        shutil.rmtree(repo_path, ignore_errors=True)
        print(f"🗑️  Cleaned up temporary directory: {repo_path}")
    except Exception as exc:
        print(f"⚠️  Could not clean up {repo_path}: {exc}")
