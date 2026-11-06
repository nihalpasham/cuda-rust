/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Mir.Launch

/-!
# Guarded 1D Kernel Access Representation (`CudaOxide.Mir.Kernel`)

Represents the buffer-access structure extracted by `lean-exporter` from a
`PostMem2Reg` `dialect-mir` kernel function:

```text
  dialect-mir CFG (PostMem2Reg)          CudaOxide.Mir.Kernel
  ──────────────────────────────────     ──────────────────────────────────────
  idx = thread::index_1d()               `globalIndex blockSize block lane`
  if idx < guardParam.len() {            `guardParam : Fin arity`
    ... load  buf[idx] ...               `AccessKind.read`  on `param : Fin arity`
    ... store buf[idx] ...               `AccessKind.write` on `param : Fin arity`
  }
```
-/
namespace CudaOxide.Mir.Kernel

/-- Direction of a buffer element operation in `dialect-mir`. -/
inductive AccessKind where
  /-- Read from a slice or buffer parameter (`mir.load` on `mir.ptr_offset`). -/
  | read
  /-- Write to a mutable slice or `DisjointSlice` (`mir.store` on `mir.ptr_offset`). -/
  | write
  deriving Repr, DecidableEq

/-- A single guarded 1D buffer access extracted from `PostMem2Reg` `dialect-mir`.
    Every access occurs under a strict dominating branch guard
    `globalIndex blockSize block lane + indexOffset < lengths guardParam`. -/
structure GuardedAccess1D (arity : Nat) where
  /-- Whether this operation loads from or stores to the target buffer. -/
  kind : AccessKind
  /-- Kernel parameter position of the accessed buffer (`0 ≤ targetParam < arity`). -/
  targetParam : Fin arity
  /-- Kernel parameter position whose `.len()` bounds the dominating branch guard. -/
  guardParam : Fin arity
  /-- Constant element offset added to the 1D thread index (`0` for `buf[idx]`). -/
  indexOffset : Nat := 0
  deriving Repr, DecidableEq

/-- Summary of a 1D kernel extracted from `dialect-mir` at `PostMem2Reg`. -/
structure KernelSummary1D (arity : Nat) where
  /-- Static X dimension of the thread block (`#[launch_contract(block = (x, 1, 1))]`). -/
  blockSize : Nat
  /-- All buffer reads and writes reachable in the kernel CFG. -/
  accesses : List (GuardedAccess1D arity)
  deriving Repr, DecidableEq

/-- Effective element index accessed by thread `(block, lane)` for `access`. -/
def GuardedAccess1D.address {arity : Nat}
    (access : GuardedAccess1D arity) (blockSize block lane : Nat) : Nat :=
  CudaOxide.Mir.globalIndex blockSize block lane + access.indexOffset

end CudaOxide.Mir.Kernel
