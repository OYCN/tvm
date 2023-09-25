import tvm
# import tvm.testing
from tvm import te
from tvm.relay.backend import Executor, Runtime
from tvm.contrib import cc
import numpy as np

tgt = tvm.target.Target(target="cuda", host="rawc")

n = te.var('n')
A = te.placeholder((n, ), name='A')
B = te.placeholder((n, ), name='B')
C = te.compute(A.shape, lambda i : A[i] + B[i], name='C')
print(type(C))

s = te.create_schedule(C.op)
bx, tx = s[C].split(C.op.axis[0], factor=64)

s[C].bind(bx, te.thread_axis('blockIdx.x'))
s[C].bind(tx, te.thread_axis('threadIdx.x'))

fadd = tvm.build(s, [A, B, C], tgt, name='myadd')

print(fadd)

with open('mod.c', 'w') as f:
  f.write(fadd.get_source())

print(fadd.imported_modules)
print(fadd.imported_modules[0])
