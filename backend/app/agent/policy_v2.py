"""Compatibility import; all policy implementation lives in policy.py."""
import sys
from . import policy
sys.modules[__name__] = policy
