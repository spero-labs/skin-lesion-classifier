#!/bin/bash

# Script to fix "too many open files" error on macOS

echo "Fixing file descriptor limits for macOS..."

# Check current limits
echo "Current soft limit: $(ulimit -n)"
echo "Current hard limit: $(ulimit -Hn)"

# Increase limits for current session
ulimit -n 4096

echo "New soft limit: $(ulimit -n)"

# For permanent fix, add to shell profile
echo ""
echo "To make this permanent, add the following line to your ~/.zshrc or ~/.bash_profile:"
echo "ulimit -n 4096"
echo ""

# System-wide fix (requires admin)
echo "For system-wide fix (requires admin password):"
echo "1. Create /Library/LaunchDaemons/limit.maxfiles.plist with:"
echo ""
cat << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>limit.maxfiles</string>
    <key>ProgramArguments</key>
    <array>
      <string>launchctl</string>
      <string>limit</string>
      <string>maxfiles</string>
      <string>65536</string>
      <string>524288</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>ServiceIPC</key>
    <false/>
  </dict>
</plist>
EOF
echo ""
echo "2. Then run:"
echo "   sudo launchctl load -w /Library/LaunchDaemons/limit.maxfiles.plist"
echo ""
echo "3. Restart your Mac"