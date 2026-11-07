/-
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-/

import CudaOxide.Mir.Launch

/-!
# Kernel Verification Contracts (`CudaOxide.Contracts`)

Defines the Lean propositions that correspond 1-to-1 with the clauses written
by a kernel author in Rust's `#[verify(...)]` attribute:

```text
  Rust `#[verify(...)]` Clause           Lean Proposition (`CudaOxide.Contracts`)
  ──────────────────────────────────     ──────────────────────────────────────────
  `no_oob_idx(buf)`                      `NoOobIdx blockSize offset guardLen bufLen`
  `disjoint_writes(buf)`                 `DisjointWrites blockSize offset`
  `writes_all_elems(buf)`                `WritesAllElems blocks blockSize outLen`
```

Each proposition states a mathematical obligation over the kernel's exported
`CudaOxide.Mir` model. Reusable proofs live in `CudaOxide.Proofs`.
-/
namespace CudaOxide.Contracts
open CudaOxide.Mir

/-- **`no_oob_idx(buf)`**: Every thread `(block, lane)` that passes the strict
    guard `globalIndex blockSize block lane < guardLen` indexes `buf` at
    `globalIndex blockSize block lane + offset < bufferLen`. -/
def NoOobIdx (blockSize offset guardLen bufferLen : Nat) : Prop :=
  ∀ block lane : Nat,                                     -- Pick ANY thread (block, lane):
    globalIndex blockSize block lane < guardLen →         -- IF it passes the branch guard (`idx < guardLen`),
    globalIndex blockSize block lane + offset < bufferLen -- THEN its buffer access is in-bounds (`< bufferLen`).

/-- **`disjoint_writes(buf)`**: Distinct launched 1D threads `(block₁, lane₁)`
    and `(block₂, lane₂)` never write to the same element index in `buf`. -/
def DisjointWrites (blockSize offset : Nat) : Prop :=
  ∀ block₁ block₂ lane₁ lane₂ : Nat,                      -- Pick ANY two threads (block₁, lane₁) and (block₂, lane₂):
    lane₁ < blockSize →                                   -- IF thread 1's lane is valid,
    lane₂ < blockSize →                                   -- AND thread 2's lane is valid,
    globalIndex blockSize block₁ lane₁ + offset =         -- AND both threads write to the exact
      globalIndex blockSize block₂ lane₂ + offset →       --     same buffer index (`idx₁ + offset = idx₂ + offset`),
    block₁ = block₂ ∧ lane₁ = lane₂                       -- THEN they are the exact same thread (`∧` = AND).

/-- **`writes_all_elems(buf)`**: Every output element index `i < outLen` in `buf`
    is written by some launched thread `(block, lane)` in the 1D grid. -/
def WritesAllElems (blocks blockSize outLen : Nat) : Prop :=
  ∀ i : Nat,                                              -- Pick ANY output element index `i`:
    i < outLen →                                          -- IF `i` is within the output length (`i < outLen`),
    ∃ block lane : Nat,                                   -- THEN there EXISTS a thread (block, lane) such that:
      block < blocks ∧                                    --   1. `block` is a valid launched block, AND
      lane < blockSize ∧                                  --   2. `lane` is a valid lane in that block, AND
      globalIndex blockSize block lane = i                --   3. that thread's global index equals `i`.

end CudaOxide.Contracts
