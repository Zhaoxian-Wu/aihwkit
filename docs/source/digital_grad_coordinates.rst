Digital gradient accumulation and write coordinates
==================================================

``AnalogContext.set_digital_grad_accumulation(True)`` incrementally forms
``grad += error.T @ input`` in digital arithmetic during each backward call.
It avoids retaining activation/error traces across accumulation micro-batches,
but does not remove the W-gradient GEMM or guarantee a speedup over deferred
batched GEMM. The optimizer must consume ``AnalogContext.grad`` explicitly;
ordinary trace-based ``AnalogSGD`` does not automatically do so.

Default weight coordinate and learnable scales
----------------------------------------------

Define ``W = diag(s) W_hat``, with s the native mapping scales (one if absent).
Native learnable output scaling is separate: ``y = diag(alpha) W x``.
The error entering AnalogFunction.backward is ``h = alpha * dL/dy``.
Default accumulation uses h unchanged: ``G = h.T @ x = dL/dW``. Alpha's
scalar or output-row-vector chain factor stays in G. Its own derivative and
the input gradient retain normal autograd semantics, including zero or signed
alpha. Additional outer operations contribute their chain factors naturally;
there is no external error-scale registration or inverse-alpha operation.

Adam or another optimizer operates on G in W coordinates, then the optimized
increment is converted once by ``delta_W_hat = delta_W / s``. Optimizer moments,
residuals and weight decay targets must all use the same coordinate. Changing
from an effective weight that includes alpha to W changes the optimizer's
parameterization; old learned-scale states require an explicit migration.

Early write preprocessing for MP-SGD
------------------------------------

Let ``ell = mapping_lr_scale``. In the absence of input-range/clipping
transformations, ``use_pre_update=True`` preprocesses h as ``h / (s * ell)``
before the digital product, giving ``G / (s * ell)``. A consumer with a matching
write-command residual and backend LR ``eta * ell`` gets the same physical
SGD increment ``-eta * G / s`` as late matrix conversion. Alpha remains in G
and is not part of the write conversion. Do not convert again at write time.

Early scaling can save a matrix pass for plain MP-SGD, but needs consistent
residual units, quantization and LR compensation. Adaptive/nonlinear operations
and clipping generally do not commute with conversion. Current project logical
optimizers, including MP-SGD, use the default ``use_pre_update=False``.
Changing this flag alone does not migrate an optimizer's coordinates.
Dynamic mapping scales need a defined update order and consistent state units.

For N samples/tokens and a Dout-by-Din tile, early error scaling costs
``O(N * Dout)`` per backward; final matrix conversion costs ``O(Dout * Din)``
per optimizer step. The product still costs ``O(N * Dout * Din)``. O(D) versus
O(D**2) applies only to one sample on a square tile; across micro-batches the
early cost can exceed a final matrix conversion. Scalar and row-vector mapping
scales obey the same algebra.

Backward independence and compatibility
---------------------------------------

Input gradients use the normal tile backward, including pre_backward, once
regardless of the accumulation flag. There is no use_pre_backward option.
With accumulation disabled, native X/D traces, alpha derivatives, and physical
updates keep their original semantics. Shared-weight inference training also
keeps its native autograd path. Runtime flags must be restored after tile load.
Indexed convolution uses the existing trace-based gradient adapter.
Input-range/clipping is outside the simple algebra above; the project's logical
matrix programming continues to reject input_range.
