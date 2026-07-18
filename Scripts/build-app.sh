#!/bin/zsh
# Distribution build: universal GaussianSplatter.app with a macOS 11.0 floor.
#
# Why this script drives swiftc directly instead of SPM/xcodebuild:
# on macOS 27 tooling (Xcode 27 / CLT 27, Swift 6.4) every higher-level build
# system clamps the macOS deployment target to 12.0:
#   - xcodebuild refuses MACOSX_DEPLOYMENT_TARGET < 12.0 outright,
#   - SwiftPM 6.4 silently raises the package's .macOS(.v11) floor to 12.0
#     (verified: even `--triple arm64-apple-macosx11.0` + old SDK emits
#     minos 12.0).
# Raw swiftc still honors -target *-apple-macos11.0, so we compile SplatCore
# as a static library and the app on top of it, then assemble the bundle by
# hand. The verification step at the bottom hard-fails if a future toolchain
# breaks the 11.0 floor, so drift is loud, not silent.
#
# Toolchain/SDK pinning:
#   - SDK: MacOSX26.2 (ships in CLT 27; supports deployment >= 10.13, and its
#     SwiftUI is pre-macro, so no Xcode macro plugins are needed).
#     The 27.0 SDK is unusable for us: minimum deployment 12.0.
#   - Toolchain: Xcode-beta if installed — the CLT 27 beta's static Swift
#     compatibility libraries dropped their x86_64 slices, which breaks
#     x86_64 linking; Xcode's copies still have them.
#
# The Xcode project in App/ is a development convenience only (12.0 floor);
# it never ships.
#
# Usage: Scripts/build-app.sh [output-dir]   (default: dist/)

set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="${1:-dist}"
APP="$OUT_DIR/GaussianSplatter.app"
VERSION="0.1"
DEPLOY="11.0"
SDK="/Library/Developer/CommandLineTools/SDKs/MacOSX26.2.sdk"
BUILD="$OUT_DIR/build"

if [[ ! -d "$SDK" ]]; then
    echo "error: pinned SDK not found: $SDK" >&2
    echo "Distribution builds need an SDK whose minimum deployment target is <= $DEPLOY." >&2
    exit 1
fi
if [[ -d /Applications/Xcode-beta.app ]]; then
    export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
fi

build_slice() {
    local arch=$1
    local target="${arch}-apple-macos${DEPLOY}"
    local bdir="$BUILD/$arch"
    mkdir -p "$bdir"

    echo "[$arch] SplatCore..."
    swiftc -target "$target" -sdk "$SDK" -O -wmo -parse-as-library \
        -emit-module -module-name SplatCore -emit-module-path "$bdir/SplatCore.swiftmodule" \
        -emit-library -static \
        Sources/SplatCore/**/*.swift \
        -o "$bdir/libSplatCore.a"

    echo "[$arch] app..."
    swiftc -target "$target" -sdk "$SDK" -O -wmo -parse-as-library \
        -I "$bdir" -L "$bdir" -lSplatCore \
        App/Sources/*.swift \
        -o "$bdir/GaussianSplatter"
}

build_slice arm64
build_slice x86_64

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

lipo -create "$BUILD/arm64/GaussianSplatter" "$BUILD/x86_64/GaussianSplatter" \
    -output "$APP/Contents/MacOS/GaussianSplatter"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleExecutable</key>
	<string>GaussianSplatter</string>
	<key>CFBundleIdentifier</key>
	<string>com.junostorti.GaussianSplatter</string>
	<key>CFBundleName</key>
	<string>GaussianSplatter</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleShortVersionString</key>
	<string>$VERSION</string>
	<key>CFBundleVersion</key>
	<string>1</string>
	<key>LSMinimumSystemVersion</key>
	<string>$DEPLOY</string>
	<key>NSHighResolutionCapable</key>
	<true/>
	<key>NSPrincipalClass</key>
	<string>NSApplication</string>
</dict>
</plist>
PLIST

printf 'APPL????' > "$APP/Contents/PkgInfo"

# Ad-hoc signature; replace with a Developer ID identity for notarized releases.
codesign --force --sign - "$APP"

echo ""
echo "== Verification =="
lipo -info "$APP/Contents/MacOS/GaussianSplatter"
fail=0
for arch in arm64 x86_64; do
    minos=$(otool -arch $arch -l "$APP/Contents/MacOS/GaussianSplatter" | awk '/LC_BUILD_VERSION/{f=1} f && /minos/{print $2; exit}')
    echo "$arch minos: $minos"
    [[ "$minos" == "$DEPLOY" ]] || fail=1
done
if [[ $fail -ne 0 ]]; then
    echo "error: a slice does not meet the $DEPLOY deployment floor — toolchain drift?" >&2
    exit 1
fi
codesign --verify --verbose=1 "$APP" && echo "codesign OK"
echo ""
echo "Built $APP"
