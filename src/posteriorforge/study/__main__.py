"""Source-checkout entry point for the additive study namespace."""
import sys
from .entrypoint import main

main(["study", *sys.argv[1:]])
