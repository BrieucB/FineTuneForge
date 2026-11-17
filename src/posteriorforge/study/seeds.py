"""Explicit namespaced streams, independent of Python's randomized hash."""
from hashlib import sha256
import numpy as np


def derive_seed(master, role, *, setup=0, case=0, bank=0, replica=0, rank=0):
    values = (master, setup, case, bank, replica, rank)
    if any(type(v) is not int or v < 0 for v in values) or not isinstance(role, str) or not role:
        raise ValueError("nonnegative integer identities and a nonempty role are required")
    tag = np.frombuffer(sha256(("hierarchical-ft-v1/" + role).encode()).digest(), dtype="<u4").tolist()
    words = np.random.SeedSequence([*values, *tag]).generate_state(2, dtype=np.uint32)
    return int(words[0]) | (int(words[1]) << 32)
