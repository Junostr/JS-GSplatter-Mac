#!/bin/bash
# CI floor verification: prove the toolchain on this machine can produce
# macOS 11.0 (Big Sur) binaries for both architectures via plain SPM, and that
# enhanced-tier (Apple-Silicon-only) code never reaches the x86_64 slice.
#
# Meant to run on a macOS 26 CI runner with Xcode 26 / Swift 6.2, where SwiftPM
# still honors the package's .macOS(.v11) platform. On a local macOS 27 machine
# (Swift 6.4) it FAILS by design — SwiftPM there clamps the deployment target
# to 12.0, which is exactly the drift this check exists to catch. Local
# distribution builds use Scripts/build-app.sh instead, which has its own
# equivalent verification.
#
# SCOPE: this proves the floor for the SwiftPM path only. Releases ship via
# Scripts/build-app.sh (raw swiftc + pinned SDK), which cannot run on a GitHub
# runner; that script self-verifies. Neither check alone covers both paths.
#
# Binary locations are queried via --show-bin-path rather than hardcoded:
# SwiftPM's layout is build-system dependent (the 6.4 "swiftbuild" system emits
# every triple to the SAME .build/out/Products/Release directory, so an x86_64
# build overwrites the arm64 one, while the older "native" system used
# per-triple .build/<triple>/release directories). Each slice is therefore
# inspected and stashed immediately after its own build, before the next arch
# can clobber it.
#
# Usage: Scripts/ci-verify-floor.sh

set -uo pipefail
cd "$(dirname "$0")/.."

EXPECTED="11.0"
PRODUCTS=(splatctl GaussianSplatterApp)
STAGE=".build/ci-slices"
FAIL=0

rm -rf "$STAGE"
mkdir -p "$STAGE"

for arch in arm64 x86_64; do
    triple="$arch-apple-macosx"
    echo "=== Building $arch ==="
    for product in "${PRODUCTS[@]}"; do
        if ! swift build -c release --triple "$triple" --product "$product"; then
            echo "error: build failed for $product ($arch)" >&2
            FAIL=1
            continue
        fi
    done

    bin=$(swift build -c release --triple "$triple" --show-bin-path 2>/dev/null)
    if [[ -z "$bin" || ! -d "$bin" ]]; then
        echo "error: could not resolve bin path for $arch" >&2
        FAIL=1
        continue
    fi

    for product in "${PRODUCTS[@]}"; do
        src="$bin/$product"
        # A missing binary must fail loudly: `nm missing | grep -c` returns 0,
        # which would otherwise sail through the x86_64==0 gate assertion below
        # and report a passing check for something that was never built.
        if [[ ! -f "$src" ]]; then
            echo "error: expected binary $src does not exist" >&2
            FAIL=1
            continue
        fi

        minos=$(otool -l "$src" | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')
        echo "$product ($arch): minos $minos"
        if [[ "$minos" != "$EXPECTED" ]]; then
            echo "error: $product ($arch) has deployment floor '$minos', expected $EXPECTED" >&2
            FAIL=1
        fi

        cp "$src" "$STAGE/$arch-$product"
    done
done

# Enhanced tier must exist only in the arm64 slice. Asymmetric assertions,
# deliberately:
#   - x86_64 count > 0 is a HARD FAILURE: enhanced-tier code in the Intel slice
#     means the #if arch(arm64) gate is broken. No optimizer setting can
#     legitimately produce this.
#   - arm64 count == 0 is only a NOTE: in an -O whole-module release build the
#     optimizer legitimately specializes and dead-strips these symbols
#     (splatctl shows 36 in debug, 0 in release, with gating fully intact).
#     Failing on that would make CI permanently red for a non-defect.
# The check is only *positively* meaningful when some product shows the
# contrast, so an all-stripped run is reported as INCONCLUSIVE rather than
# being allowed to masquerade as a pass.
echo "=== Enhanced-tier arch gating ==="
gate_demonstrated=0
for product in "${PRODUCTS[@]}"; do
    a_bin="$STAGE/arm64-$product"
    x_bin="$STAGE/x86_64-$product"
    [[ -f "$a_bin" && -f "$x_bin" ]] || { echo "note: $product missing a slice, skipping gate check"; continue; }

    enh_arm64=$(nm "$a_bin" 2>/dev/null | grep -c EnhancedTrainingEngine)
    enh_x86=$(nm "$x_bin" 2>/dev/null | grep -c EnhancedTrainingEngine)
    echo "$product EnhancedTrainingEngine symbols: arm64=$enh_arm64 x86_64=$enh_x86"

    if [[ "$enh_x86" -ne 0 ]]; then
        echo "error: $product x86_64 slice contains enhanced-tier symbols — arch gating is broken" >&2
        FAIL=1
    fi
    if [[ "$enh_arm64" -gt 0 ]]; then
        gate_demonstrated=1
    else
        echo "note: $product arm64 slice has no enhanced-tier symbols (optimizer stripped them); gating unverifiable via this product"
    fi
done
if [[ "$gate_demonstrated" -eq 0 ]]; then
    echo "warning: no product retained arm64 enhanced-tier symbols — arch-gate check was INCONCLUSIVE, not passing" >&2
fi

# Universal executable for inspection, assembled from the stashed slices.
if [[ -f "$STAGE/arm64-GaussianSplatterApp" && -f "$STAGE/x86_64-GaussianSplatterApp" ]]; then
    mkdir -p dist
    lipo -create "$STAGE/arm64-GaussianSplatterApp" "$STAGE/x86_64-GaussianSplatterApp" \
        -output dist/GaussianSplatter-universal
    lipo -info dist/GaussianSplatter-universal
fi

if [[ $FAIL -ne 0 ]]; then
    echo "FAILED: deployment floor or arch gating regressed." >&2
else
    echo "OK: both slices at minos $EXPECTED, no enhanced-tier symbols in x86_64."
fi
exit $FAIL
