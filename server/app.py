import os
import sys

# Ensure the root directory is on the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keepalive import main as keepalive_main

def main():
    """
    OpenEnv server entry point.
    Runs the keepalive server which satisfies standard POST/GET requirements.
    """
    keepalive_main()

if __name__ == "__main__":
    main()
