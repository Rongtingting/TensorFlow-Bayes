Notes
=====

Issues with MacBook
-------------------

MacBook fails to work with both tf.tensordot and np.tensordot (or np.dot),
while Linux works fine.

Both: Python 3.7.5; numpy 1.7.3; tensorflow 2.0.0


import numpy as np
import tensorflow as tf

Nd, Nk, Ns = 3, 2, 5
w_np = np.ones([Nd, Nk], dtype=np.float32)
z_np = np.ones([Nk, Ns], dtype=np.float32)

tf.tensordot(w_np, z_np, axes=1)
np.tensordot(w_np, z_np, axes=1)