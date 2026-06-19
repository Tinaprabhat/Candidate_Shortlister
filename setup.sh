#!/bin/bash
###############################################################################
# setup.sh — Decompress all model artifacts after cloning.
# Run once after `git clone` (and `git lfs pull`).
###############################################################################
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMP="$DIR/models/compressed"
DECOMP="$DIR/models/decompressed"

echo "==============================================="
echo " RedRob — Model Setup"
echo "==============================================="

if [ ! -d "$COMP" ]; then
  echo "❌ $COMP not found. Did you clone with Git LFS?"
  echo "   git lfs install && git lfs pull"
  exit 1
fi

mkdir -p "$DECOMP"

echo "📦 Decompressing artifacts..."
shopt -s nullglob
archives=("$COMP"/*.tar.gz)
if [ ${#archives[@]} -eq 0 ]; then
  echo "⚠️  No .tar.gz archives found in $COMP."
  echo "   Run: python build_kb.py --all   (one-time, then commit archives)"
  exit 1
fi

for archive in "${archives[@]}"; do
  name="$(basename "$archive")"
  echo "  → extracting $name"
  tar -xzf "$archive" -C "$DECOMP" &
done
wait

# Verify checksums if present
if [ -f "$COMP/checksums.sha256" ]; then
  echo "🔐 Verifying checksums..."
  ( cd "$COMP" && sha256sum -c checksums.sha256 >/dev/null 2>&1 \
      && echo "  ✅ checksums OK" \
      || echo "  ⚠️  checksum mismatch (continuing anyway)" )
fi

echo ""
echo "✅ Setup complete. Artifacts in $DECOMP"
echo "Next: python setup_check.py"
