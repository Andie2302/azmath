#!/usr/bin/env python3
"""Offline distribution references. Requires mpmath 1.3.0; cargo needs no Python."""
import struct
from pathlib import Path
import mpmath as mp
from reference_ieee import bits


def lower(name, x, a, b):
    if name == 'normal': return mp.erfc(-x/mp.sqrt(2))/2
    if name == 'exponential': return -mp.expm1(-a*x) if x > 0 else mp.mpf(0)
    if name == 'uniform': return min(mp.mpf(1), max(mp.mpf(0), (x-a)/(b-a)))
    if name == 'gamma': return mp.gammainc(a, 0, x/b, regularized=True)
    if name == 'chi2': return mp.gammainc(a/2, 0, x/2, regularized=True)
    if name == 'beta': return mp.betainc(a, b, 0, x, regularized=True)
    if name == 'student_t': return 1-upper(name, x, a, b) if x >= 0 else upper(name, -x, a, b)
    z=a*x/(b+a*x)
    return mp.betainc(a/2, b/2, 0, z, regularized=True)


def upper(name, x, a, b):
    if name == 'normal': return mp.erfc(x/mp.sqrt(2))/2
    if name == 'exponential': return mp.exp(-a*x) if x > 0 else mp.mpf(1)
    if name == 'uniform': return min(mp.mpf(1), max(mp.mpf(0), (b-x)/(b-a)))
    if name == 'gamma': return mp.gammainc(a, x/b, mp.inf, regularized=True)
    if name == 'chi2': return mp.gammainc(a/2, x/2, mp.inf, regularized=True)
    if name == 'beta': return mp.betainc(b, a, 0, 1-x, regularized=True)
    if name == 'student_t':
        t=mp.betainc(a/2, mp.mpf('.5'), 0, a/(a+x*x), regularized=True)/2
        return t if x >= 0 else 1-t
    z=b/(b+a*x)
    return mp.betainc(b/2, a/2, 0, z, regularized=True)


def positive_inverse(name, q, a, b, is_upper):
    # Work in log(x) so lower tails such as Gamma(.25) at p=1e-100
    # are resolved by relative precision, not an absolute x tolerance.
    lo=mp.mpf(-4000); hi=mp.mpf(0 if name == 'beta' else 8 if name == 'normal' else 4000)
    for _ in range(240):
        m=(lo+hi)/2; x=mp.exp(m)
        v=upper(name,x,a,b) if is_upper else lower(name,x,a,b)
        if (v > q) if is_upper else (v < q): lo=m
        else: hi=m
    return mp.exp((lo+hi)/2)


def quantile(name, p, a, b, isf):
    if name == 'uniform': return p*a+(1-p)*b if isf else (1-p)*a+p*b
    if p == 0: return (mp.mpf(1) if name == 'beta' else mp.inf) if isf else -mp.inf if name in ['normal','student_t'] else mp.mpf(0)
    if p == 1: return -mp.inf if isf and name in ['normal','student_t'] else mp.mpf(0) if isf else mp.mpf(1) if name=='beta' else mp.inf
    if name == 'exponential': return -mp.log(p)/a if isf else -mp.log1p(-p)/a
    if name == 'f' and a == 2 and b == 2: return (1-p)/p if isf else p/(1-p)
    if name in ['normal','student_t']:
        if p == mp.mpf('.5'): return mp.mpf(0)
        x=positive_inverse(name,min(p,1-p),a,b,True)
        negative=(p > mp.mpf('.5')) if isf else (p < mp.mpf('.5'))
        return -x if negative else x
    # Always evaluate the smaller probability directly.
    q = p if p <= mp.mpf('.5') else 1-p
    return positive_inverse(name,q,a,b,isf == (p <= mp.mpf('.5')))


families=[('normal',0,1),('exponential',2,0),('uniform',-2,3),
          ('gamma',.25,1.5),('gamma',2.5,1.5),('chi2',5,0),
          ('beta',.25,3),('beta',2,5),('student_t',1,0),('student_t',10,0),
          ('f',2,2),('f',3,7)]
rows=[]
for width in [32,64]:
    fmt,integ=('f','I') if width==32 else ('d','Q')
    def exact(v):
        packed=struct.pack('>'+fmt,float(v)); f=struct.unpack('>'+fmt,packed)[0]
        n,d=f.as_integer_ratio()
        return n,d,struct.unpack('>'+integ,packed)[0]
    for name,av,bv in families:
        print(f"f{width}: {name}({av},{bv})",flush=True)
        xs=([-38,-10,-2,0,2,8,10,38] if name=='normal' else
            [-2,0,1,3] if name=='uniform' else
            [1e-30,.1,.9] if name=='beta' else [1e-30,.1,1,10,100])
        if name in ['student_t','f']: xs += [1e200]
        if name == 'gamma': xs += [1e-200,1000]
        cases=[(kind,x) for kind in ['cdf','sf'] for x in xs]
        ps=[1e-100,1e-30,.01,.5,.99,0.9999999403953552,0.9999999999999999]
        if name in ['normal','student_t','gamma','f']: ps += [1e-300]
        cases += [(kind,p) for kind in ['quantile','isf'] for p in ps]
        seen=set()
        for kind,value in cases:
            # Skip finite f64 arguments beyond the f32 range.
            if width==32 and abs(float(value))>3.4028234663852886e38: continue
            xn,xd,xb=exact(value);an,ad,ab=exact(av);bn,bd,bb=exact(bv)
            if (kind,xb) in seen: continue
            seen.add((kind,xb)); answers=[]
            for precision in [100,200]:
                with mp.workdps(precision):
                    x=mp.mpf(xn)/xd; a=mp.mpf(an)/ad; b=mp.mpf(bn)/bd
                    if kind in ['quantile','isf']: result=quantile(name,x,a,b,kind=='isf')
                    else: result=(upper if kind=='sf' else lower)(name,x,a,b)
                    answers.append(bits(result,width))
            assert answers[0]==answers[1],(name,kind,width,value,answers)
            rows.append(f'    ("{name}_{kind}", {width}, 0x{ab:016x}, 0x{bb:016x}, 0x{xb:016x}, 0x{answers[0]:016x}),')
header=f'// mpmath {mp.__version__}; scripts/generate_dist_refs.py.\n'
header+='// Direct IEEE rounding agrees at 100 and 200 decimal digits.\n'
header+='#[rustfmt::skip]\n'
header+='pub const CASES: &[(&str,u32,u64,u64,u64,u64)] = &[\n'
Path('tests/data/dist_refs.rs').write_text(header+'\n'.join(rows)+'\n];\n')
print(f'Generated {len(rows)} distribution vectors')
