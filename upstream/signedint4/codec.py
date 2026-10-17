"""Exact low-nibble-first offset/two's-complement codecs. No quantization."""
from __future__ import annotations


def _nibble(value: int) -> int:
    if type(value) is not int or not 0 <= value < 16:
        raise ValueError('nibble must be an integer in [0,15]')
    return value


def offset_decode(code: int) -> int:
    return _nibble(code) - 8


def signed_decode(code: int) -> int:
    code = _nibble(code)
    return code - 16 if code >= 8 else code


def recode_nibble(code: int) -> int:
    return _nibble(code) ^ 8


def recode_bytes(packed: bytes) -> bytes:
    if not isinstance(packed, (bytes, bytearray, memoryview)):
        raise TypeError('expected a bytes-like packed state')
    return bytes(b ^ 0x88 for b in bytes(packed))


def unpack(packed: bytes, *, signed: bool = False) -> tuple[int, ...]:
    if not isinstance(packed, (bytes, bytearray, memoryview)):
        raise TypeError('expected a bytes-like packed state')
    decode = signed_decode if signed else offset_decode
    return tuple(decode(c) for b in bytes(packed) for c in (b & 15, b >> 4))


def pack(values, *, signed: bool = False) -> bytes:
    values = list(values)
    if len(values) % 2:
        raise ValueError('an even number of coordinates is required')
    if any(type(q) is not int or not -8 <= q <= 7 for q in values):
        raise ValueError('represented integers must be in [-8,7]')
    codes = [(q & 15) if signed else (q + 8) for q in values]
    return bytes(codes[i] | codes[i + 1] << 4 for i in range(0, len(codes), 2))
