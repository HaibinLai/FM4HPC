"""Exact model of optimi's Kahan-compensated low-precision parameter update (single-tensor path, bf16).

Source (pinned optimi 17a3d54, optimi/adam.py `_single_param_adam`, kahan_sum and param.dtype in [fp16, bf16]):
    kahan_comp.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-lr)    # c <- rn(c + (-lr*m)/denom)
    grad.copy_(param.detach())                                              # t <- w
    param.add_(kahan_comp)                                                  # w <- rn(w + c)
    kahan_comp.add_(grad.sub_(param))                                       # c <- rn(c + rn(t - w))
All tensors bf16. On CPU PyTorch computes each op in float32 and rounds ONCE to bf16 (round-to-nearest-even);
this module emulates exactly that with numpy and is validated bit-for-bit against torch (validate_kahan_model.py).
"""
import numpy as np

def to_bf16(x):
    """float32 array/scalar -> nearest-even bf16, returned as float32 (same shape) holding the bf16 value."""
    shape = np.shape(x)
    x = np.atleast_1d(np.asarray(x, dtype=np.float32)).copy()
    u = x.view(np.uint32).astype(np.uint64)
    nan = np.isnan(x)
    bias = np.uint64(0x7FFF) + ((u >> np.uint64(16)) & np.uint64(1))
    r = ((u + bias) & np.uint64(0xFFFF0000)).astype(np.uint32)
    out = r.view(np.float32).copy()
    out[nan] = np.nan
    return out.reshape(shape)

def bf16_add(a, b): return to_bf16(np.float32(a) + np.float32(b))
def bf16_sub(a, b): return to_bf16(np.float32(a) - np.float32(b))

def addcdiv_value(c, m, d, value, variant):
    """c + value*m/d in float32 then rn to bf16.  variant 'a': (value*m)/d ; 'b': value*(m/d)."""
    c, m, d, v = np.float32(c), np.float32(m), np.float32(d), np.float32(value)
    inner = (v * m) / d if variant == 'a' else v * (m / d)
    return to_bf16(c + inner)

def kahan_step(w, c, m, v, lr, eps, variant='a'):
    """One Kahan-compensated update of (w, c) given post-update moments m (exp_avg) and v (exp_avg_sq); all bf16 values."""
    denom = bf16_add(to_bf16(np.sqrt(np.float32(v))), np.float32(eps))        # exp_avg_sq.sqrt().add_(eps)
    c1 = addcdiv_value(c, m, denom, -np.float32(lr), variant)                  # c <- c - lr*m/denom
    t = w                                                                      # grad.copy_(param)
    w1 = bf16_add(w, c1)                                                       # param.add_(kahan_comp)
    d = bf16_sub(t, w1)                                                        # grad.sub_(param)
    c2 = bf16_add(c1, d)                                                       # kahan_comp.add_(...)
    return w1, c2

def naive_step(w, m, v, lr, eps, variant='a'):
    """optimi without Kahan: param.addcdiv_(exp_avg, sqrt(v)+eps, value=-lr), all in bf16."""
    denom = bf16_add(to_bf16(np.sqrt(np.float32(v))), np.float32(eps))
    return addcdiv_value(w, m, denom, -np.float32(lr), variant)
