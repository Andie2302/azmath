"""Shared direct IEEE rounding for offline mpmath reference generators."""
import mpmath as mp


def bits(v, width):
    """Round high precision to nearest IEEE value, ties to even (no double rounding)."""
    if mp.isnan(v):
        raise ValueError('NaN reference')
    if mp.isinf(v):
        return ((int(v < 0) << (width - 1))
                | (0x7f800000 if width == 32 else 0x7ff0000000000000))
    fb, eb = (23, 8) if width == 32 else (52, 11)
    sign = int(v < 0)
    v = abs(v)
    if not v:
        return 0
    bias = (1 << (eb - 1)) - 1
    e = max(int(mp.floor(mp.log(v, 2))), 1 - bias)
    scaled = v / mp.power(2, e - fb)
    m = int(mp.floor(scaled))
    tail = scaled - m
    m += int(tail > mp.mpf('0.5') or (tail == mp.mpf('0.5') and m % 2))
    if m >= 1 << (fb + 1):
        m >>= 1
        e += 1
    if e > bias:
        return (sign << (width - 1)) | (((1 << eb) - 1) << fb)
    exponent = 0 if m < 1 << fb else e + bias
    fraction = m if exponent == 0 else m - (1 << fb)
    return (sign << (width - 1)) | (exponent << fb) | fraction
