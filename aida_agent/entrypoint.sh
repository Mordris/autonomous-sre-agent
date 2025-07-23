#!/bin/bash
set -e

# Take ownership of the Hugging Face cache directory.
# This ensures the non-root 'aida' user can write to the mounted volume.
sudo chown -R aida:aida /home/aida/.cache

# Execute the command passed to this script (which will be our CMD from the Dockerfile)
exec "$@"