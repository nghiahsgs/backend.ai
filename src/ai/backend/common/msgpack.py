"""
Wrapper of msgpack-python with good defaults.
"""

import datetime
import enum
import os
import pickle
import uuid
from decimal import Decimal
from pathlib import PosixPath, PurePosixPath
from typing import Any

import msgpack as _msgpack
import temporenc

from .types import BinarySize, ResourceSlot

__all__ = ("packb", "unpackb")


class ExtTypes(enum.IntEnum):
    # We can define up to 128 extension type identifiers.
    UUID = 1
    DATETIME = 2
    DECIMAL = 3
    POSIX_PATH = 4
    PURE_POSIX_PATH = 5
    ENUM = 6
    RESOURCE_SLOT = 8
    BACKENDAI_BINARY_SIZE = 16


def _default(obj: object) -> _msgpack.ExtType:
    match obj:
        case tuple():
            return list(obj)
        case uuid.UUID():
            return _msgpack.ExtType(ExtTypes.UUID, obj.bytes)
        case datetime.datetime():
            return _msgpack.ExtType(ExtTypes.DATETIME, temporenc.packb(obj))
        case BinarySize():
            return _msgpack.ExtType(ExtTypes.BACKENDAI_BINARY_SIZE, pickle.dumps(obj, protocol=5))
        case Decimal():
            return _msgpack.ExtType(ExtTypes.DECIMAL, pickle.dumps(obj, protocol=5))
        case PosixPath():
            return _msgpack.ExtType(ExtTypes.POSIX_PATH, os.fsencode(obj))
        case PurePosixPath():
            return _msgpack.ExtType(ExtTypes.PURE_POSIX_PATH, os.fsencode(obj))
        case ResourceSlot():
            return _msgpack.ExtType(ExtTypes.RESOURCE_SLOT, pickle.dumps(obj, protocol=5))
        case enum.Enum():
            return _msgpack.ExtType(ExtTypes.ENUM, pickle.dumps(obj, protocol=5))
    raise TypeError(f"Unknown type: {obj!r} ({type(obj)})")


def _ext_hook(code: int, data: bytes) -> Any:
    match code:
        case ExtTypes.UUID:
            return uuid.UUID(bytes=data)
        case ExtTypes.DATETIME:
            return temporenc.unpackb(data).datetime()
        case ExtTypes.DECIMAL:
            return pickle.loads(data)
        case ExtTypes.POSIX_PATH:
            return PosixPath(os.fsdecode(data))
        case ExtTypes.PURE_POSIX_PATH:
            return PurePosixPath(os.fsdecode(data))
        case ExtTypes.ENUM:
            return pickle.loads(data)
        case ExtTypes.RESOURCE_SLOT:
            return pickle.loads(data)
        case ExtTypes.BACKENDAI_BINARY_SIZE:
            return pickle.loads(data)
    return _msgpack.ExtType(code, data)


DEFAULT_PACK_OPTS = {
    "use_bin_type": True,  # bytes -> bin type (default for Python 3)
    "strict_types": True,  # do not serialize subclasses using superclasses
    "default": _default,
}

DEFAULT_UNPACK_OPTS = {
    "raw": False,  # assume str as UTF-8 (default for Python 3)
    "strict_map_key": False,  # allow using UUID as map keys
    "use_list": False,  # array -> tuple
    "ext_hook": _ext_hook,
}


def packb(data: Any, **kwargs) -> bytes:
    opts = {**DEFAULT_PACK_OPTS, **kwargs}
    ret = _msgpack.packb(data, **opts)
    if ret is None:
        return b""
    return ret


def unpackb(packed: bytes, **kwargs) -> Any:
    opts = {**DEFAULT_UNPACK_OPTS, **kwargs}
    return _msgpack.unpackb(packed, **opts)
