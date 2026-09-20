"""Create a local recovery credential once; never overwrite existing settings."""
from pathlib import Path
import secrets

root = Path(__file__).resolve().parents[2]
path = root / ".env.recovery"
if path.exists():
    print("Existing .env.recovery retained")
else:
    with path.open("x", encoding="utf-8") as fp:
        fp.write(f"RECOVERY_DB_PASSWORD={secrets.token_urlsafe(32)}\n")
        fp.write("RECOVERY_DB_PORT=55432\nTHESIS_ROOT=..\n")
    path.chmod(0o600)
    print("Created .env.recovery (credential not printed)")
