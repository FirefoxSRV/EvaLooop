"""Cross-cycle anchor strategies.

Importing this package registers the built-in anchors. To add one, create a
BaseAnchor subclass decorated with @register_anchor("your_name") and import it
below.
"""

from .base import BaseAnchor, create_anchor, list_anchors, register_anchor
from .strategies import (
    AssertAnchor,
    MaskedAssertAnchor,
    SignatureAnchor,
    entry_point_from_anchor,
    mask_assert_oracle,
    signature_from_code,
)

__all__ = [
    "BaseAnchor",
    "create_anchor",
    "list_anchors",
    "register_anchor",
    "AssertAnchor",
    "MaskedAssertAnchor",
    "SignatureAnchor",
    "mask_assert_oracle",
    "signature_from_code",
    "entry_point_from_anchor",
]
