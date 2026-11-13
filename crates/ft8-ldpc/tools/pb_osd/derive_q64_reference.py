#!/usr/bin/env python3
"""Regenerate independent 90-digit references for the shared f64 Q interface."""
import importlib.util
from pathlib import Path
import random
import struct
import sys

sys.dont_write_bytecode=True
root=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('derive_q',Path(__file__).with_name('derive_q.py'))
reference=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
D=reference.D
points=[14*i/2048 for i in range(2049)]
points += [14+26*i/2048 for i in range(2049)]
rng=random.Random(0x51463634524546)
points += [rng.uniform(0,40) for _ in range(256)]
points += [0.75,1.,3.,6.,7.,8.,10.,14.,16.,20.,26.,30.,36.,37.,38.,38.4,38.46,38.47,38.5,40.]
for boundary in [i/2 for i in range(2,22)]+[28.]:
    x=boundary*float(D(2).sqrt())
    bits=struct.unpack('<Q',struct.pack('<d',x))[0]
    points += [struct.unpack('<d',struct.pack('<Q',b))[0] for b in range(bits-4,bits+5)]
path=Path(__file__).with_name('q64_reference.bin')
with path.open('wb') as out:
    for x in sorted(set(points)):
        y=D(x)/D(2).sqrt()
        q=reference.scaled_tail(y)*(-y*y).exp()
        out.write(struct.pack('<3d',x,float(q),float(1-q)))
print('wrote',len(set(points)),'f64 reference records')
