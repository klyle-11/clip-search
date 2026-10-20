"""Settings from the environment / .env. Every CLIPSEARCH_X also accepts the legacy JBP_X name."""
import os

from dotenv import load_dotenv

load_dotenv()

LEGACY = {"RSS": "JBP_PATREON_RSS"}


def env(name: str, default: str | None = None) -> str | None:
    for key in (f"CLIPSEARCH_{name}", LEGACY.get(name, f"JBP_{name}")):
        if os.getenv(key):
            return os.getenv(key)
    return default
