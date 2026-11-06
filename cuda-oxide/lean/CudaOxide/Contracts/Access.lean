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
  ∀ block lane : Nat,
    globalIndex blockSize block lane < guardLen →
    globalIndex blockSize block lane + offset < bufferLen

/-- **`disjoint_writes(buf)`**: Distinct launched 1D threads `(block₁, lane₁)`
    and `(block₂, lane₂)` never write to the same element index in `buf`. -/
def DisjointWrites (blockSize offset : Nat) : Prop :=
  ∀ block₁ block₂ lane₁ lane₂ : Nat,
    lane₁ < blockSize →
    lane₂ < blockSize →
    globalIndex blockSize block₁ lane₁ + offset =
      globalIndex blockSize block₂ lane₂ + offset →
    block₁ = block₂ ∧ lane₁ = lane₂

/-- **`writes_all_elems(buf)`**: Every output element index `i < outLen` in `buf`
    is written by some launched thread `(block, lane)` in the 1D grid. -/
def WritesAllElems (blocks blockSize outLen : Nat) : Prop :=
  ∀ i : Nat,
    i < outLen →
    ∃ block lane : Nat,
      block < blocks ∧
      lane < blockSize ∧
      globalIndex blockSize block lane = i

end CudaOxide.Contracts
