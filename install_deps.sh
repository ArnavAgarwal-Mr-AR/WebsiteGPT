```bash
#!/bin/bash
set -e

# Install system dependencies for Playwright
apt-get update
apt-get install -y \
    libgtk-4-1 \
    libgraphene-1.0-0 \
    libgstreamer-gl1.0-0 \
    libgstreamer-plugins-bad1.0-0 \
    libavif15 \
    libenchant-2-2 \
    libsecret-1-0 \
    libmanette-0.2-0 \
    libgles2

# Clean up to reduce image size
rm -rf /var/lib/apt/lists/*

# Install Python dependencies
pip install -r requirements.txt

# Install Playwright browsers
/opt/render/project/src/.venv/bin/playwright install

# Run crawl4ai setup
/opt/render/project/src/.venv/bin/crawl4ai-setup
```