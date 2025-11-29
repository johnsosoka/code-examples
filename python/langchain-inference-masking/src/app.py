"""
Entry point for PII masking middleware demos.

This module provides a simple CLI interface to run different demo scenarios:
- Default (no args): Simple PII masking demo without tools
- verify: PII masking with identity verification tool integration
"""

import sys


def main():
    """Run the appropriate demo based on CLI arguments."""
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        from agents.verification_agent import run_verification_demo
        run_verification_demo()
    else:
        from agents.simple_agent import run_simple_demo
        run_simple_demo()


if __name__ == "__main__":
    main()
