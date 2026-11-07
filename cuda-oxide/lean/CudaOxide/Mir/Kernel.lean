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

/-- Memory address space of a buffer access. -/
inductive MemorySpace where
  /-- Global device memory (`&[T]`, `DisjointSlice`). -/
  | global
  /-- Per-block shared memory (`SharedArray<T, N>`). -/
  | shared
  deriving Repr, DecidableEq

/-- 2D shared-memory tile indexing pattern inside a `(blockDimX, blockDimY, 1)` block. -/
inductive TileAccessPattern where
  /-- Unique per-thread staging slot `ty * stride + tx`. -/
  | threadSlot (stride : Nat)
  /-- Row scan `ty * stride + i` across loop index `i`. -/
  | rowScan (stride : Nat)
  /-- Column scan `i * stride + tx` down loop index `i`. -/
  | colScan (stride : Nat)
  deriving Repr, DecidableEq

/-- Evaluate the flat element index of a 2D tile access for thread `(tx, ty)` at step `i`. -/
def TileAccessPattern.address (pat : TileAccessPattern) (tx ty i : Nat) : Nat :=
  match pat with
  | .threadSlot stride => CudaOxide.Mir.globalIndex stride ty tx
  | .rowScan stride => CudaOxide.Mir.globalIndex stride ty i
  | .colScan stride => CudaOxide.Mir.globalIndex stride i tx

/-- A shared-memory tile access tagged with its barrier phase (`0`, `1`, ...). -/
structure PhasedTileAccess where
  /-- Read or write access. -/
  kind : AccessKind
  /-- Memory space (`.shared` for `SharedArray`). -/
  space : MemorySpace := .shared
  /-- Identifier of the target buffer (`0` for `TILE_A`, `1` for `TILE_B`, ...). -/
  bufferId : Nat
  /-- Barrier phase index (`0` before first `sync_threads()`, `1` after first, etc.). -/
  phase : Nat
  /-- 2D coordinate formula used to index the buffer. -/
  pattern : TileAccessPattern
  /-- Exclusive upper bound on loop index `i` (`1` if not inside an inner scan loop). -/
  loopBound : Nat := 1
  deriving Repr, DecidableEq

/-- Summary of a 2D phased shared-memory kernel extracted at `PostMem2Reg`. -/
structure TiledKernelSummary2D where
  /-- Static X dimension of the thread block (`reqntid_x`). -/
  blockDimX : Nat
  /-- Static Y dimension of the thread block (`reqntid_y`). -/
  blockDimY : Nat
  /-- Number of `sync_threads()` barriers in the tile loop. -/
  numBarriers : Nat
  /-- Whether all barriers execute in thread-uniform control flow. -/
  barriersUniform : Bool
  /-- All phased shared-memory accesses extracted from the kernel CFG. -/
  sharedAccesses : List PhasedTileAccess
  deriving Repr, DecidableEq

end CudaOxide.Mir.Kernel
