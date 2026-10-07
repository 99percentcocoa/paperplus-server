import sys
from pathlib import Path

# Allow `import shared.contracts` from the sibling shared/ package at the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
