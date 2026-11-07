/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed SSA lookup table (`SsaEnv`).
//!
//! Instead of mapping a Pliron SSA [`Value`] to an untyped `String`, every SSA
//! value is bound to a [`TypedSsa`] carrying **both**:
//! 1. `lean_name`: its generated Lean variable name (`"arg0"`, `"v0"`, `"v1"`, ...)
//! 2. `kind`: its MIR type and semantic role ([`SsaKind`])
//!
//! ```text
//!   Pliron `Value`      `TypedSsa { lean_name, kind }`
//!   ───────────────     ────────────────────────────────────────────────────
//!   `block` (arg 0) ──► `{ lean_name: "arg0", kind: ScalarWord { width: 64 } }`
//!   `a`     (arg 0) ──► `{ lean_name: "arg0", kind: BufferParam(0) }`
//!   `prod`  (op 0)  ──► `{ lean_name: "v0",   kind: ScalarWord { width: 64 } }`
//!   `idx`   (op 1)  ──► `{ lean_name: "v1",   kind: Index1D { offset: 0 } }`
//!   `c_len` (op 2)  ──► `{ lean_name: "v2",   kind: SliceLen(2) }`
//!   `cond`  (op 3)  ──► `{ lean_name: "v3",   kind: IndexLtLen { guard_param: 2 } }`
//! ```

use std::collections::HashMap;

use pliron::value::Value;

use crate::{ExportError, reject};

/// 2D shared-memory tile coordinate shape inside a `(blockDimX, blockDimY, 1)` block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TileAccessPatternKind {
    /// Per-thread unique slot `ty * stride + tx`.
    ThreadSlot {
        /// Row stride of the shared tile.
        stride: u64,
    },
    /// Row scan `ty * stride + i` across loop step `i`.
    RowScan {
        /// Row stride of the shared tile.
        stride: u64,
    },
    /// Column scan `i * stride + tx` down loop step `i`.
    ColScan {
        /// Row stride of the shared tile.
        stride: u64,
    },
}

/// Semantic type and provenance of an SSA value in `dialect-mir`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SsaKind {
    /// Fixed-width unsigned integer word (`ui8`, `ui16`, `ui32`, `ui64`).
    ScalarWord {
        /// Bit width (`8`, `16`, `32`, or `64`).
        width: u32,
    },
    /// Integer constant literal with known unsigned value.
    ConstInt {
        /// Bit width (`0` when signless or boolean).
        width: u32,
        /// Unsigned literal value.
        value: u128,
    },
    /// Entry block slice or `DisjointSlice` parameter at `param` (`0..arity`).
    BufferParam(usize),
    /// Stack alloca for the zero-sized `LaunchContext` token.
    LaunchContextAlloca,
    /// `ThreadIndex` struct returned by `__internal::index_1d`.
    ThreadIndexStruct,
    /// 1D global thread index plus a constant element offset (`idx + offset`).
    Index1D {
        /// Constant offset added to `thread::index_1d()`.
        offset: u64,
    },
    /// Checked addition tuple `(Index1D + offset, overflow_bool)`.
    CheckedIndex1D {
        /// Constant offset added to `thread::index_1d()`.
        offset: u64,
    },
    /// Base pointer extracted from buffer parameter `param` (`field 0`).
    SlicePtr(usize),
    /// Element length extracted from buffer parameter `param` (`field 1`).
    SliceLen(usize),
    /// Pointer to `&buf[idx + offset]` on buffer parameter `param`.
    ElementPtr {
        /// Target buffer parameter index.
        param: usize,
        /// Constant element offset added to the 1D thread index.
        offset: u64,
    },
    /// Boolean comparison `idx == usize::MAX` (sentinel check).
    IsSentinel,
    /// Boolean comparison `idx < param.len()`.
    IndexLtLen {
        /// Buffer parameter whose `.len()` bounds `idx`.
        guard_param: usize,
    },
    /// Boolean negation `!(idx < param.len())`.
    IndexGeLen {
        /// Buffer parameter whose `.len()` bounds `idx`.
        guard_param: usize,
    },
    /// `Option<&mut T>` carrying `None`.
    OptionNone,
    /// `Option<&mut T>` carrying `Some(&mut param[idx + offset])` constructed
    /// inside a block guarded by `idx < guard_param.len()`.
    OptionSome {
        /// Target buffer parameter index.
        param: usize,
        /// Constant element offset.
        offset: u64,
        /// Dominating `< len` guard parameter index.
        guard_param: usize,
    },
    /// Merged `Option<&mut T>` block argument at a join block after `get_mut`.
    OptionJoin {
        /// Target buffer parameter index.
        param: usize,
        /// Constant element offset.
        offset: u64,
        /// Dominating `< len` guard parameter index from the `Some` branch.
        guard_param: usize,
    },
    /// Discriminant extracted from `OptionJoin`.
    OptionDiscriminant {
        /// Target buffer parameter index.
        param: usize,
        /// Constant element offset.
        offset: u64,
        /// Dominating `< len` guard parameter index from the `Some` branch.
        guard_param: usize,
    },
    /// Boolean test `discriminant == 1` (`Option::is_some`).
    OptionIsSome {
        /// Target buffer parameter index.
        param: usize,
        /// Constant element offset.
        offset: u64,
        /// Dominating `< len` guard parameter index.
        guard_param: usize,
    },
    /// Boolean test `discriminant == 0` (`Option::is_none`).
    OptionIsNone,
    /// 2D block thread index `threadIdx.x` (`0 <= tx < blockDim.x`).
    TidX,
    /// 2D block thread index `threadIdx.y` (`0 <= ty < blockDim.y`).
    TidY,
    /// Inner tile scan loop counter `i` bounded by `0 <= i < bound`.
    LoopStep {
        /// Exclusive upper bound on `i` from `while i < bound`.
        bound: u64,
    },
    /// Stride product `ty * stride` or `i * stride` in a 2D tile coordinate.
    TileRowScaled {
        /// Whether the row term is `TidY` (`None`) or `LoopStep { bound }` (`Some(bound)`).
        loop_bound: Option<u64>,
        /// Constant row stride.
        stride: u64,
    },
    /// Linear 2D shared-memory tile index (`ty * W + tx`, `ty * W + i`, or `i * W + tx`).
    TileCoord2D {
        /// 2D indexing pattern.
        pattern: TileAccessPatternKind,
        /// Exclusive upper bound on `i` (`1` for `ThreadSlot`).
        loop_bound: u64,
    },
    /// Base pointer to a static `SharedArray<T, N>` allocation.
    SharedBasePtr {
        /// Index of the `SharedArray` static (`0`, `1`, ...).
        shared_id: usize,
        /// Number of elements in the `SharedArray`.
        size: u64,
    },
    /// Boolean bounds test `tile_coord < size` on a `SharedArray`.
    SharedIndexLtLen {
        /// Constant element count compared against.
        size: u64,
        /// 2D indexing pattern.
        pattern: TileAccessPatternKind,
        /// Exclusive upper bound on `i`.
        loop_bound: u64,
    },
    /// Element pointer `&TILE[tile_coord]` into a `SharedArray`.
    SharedElementPtr {
        /// Index of the `SharedArray` static (`0`, `1`, ...).
        shared_id: usize,
        /// Number of elements in the `SharedArray`.
        size: u64,
        /// 2D indexing pattern.
        pattern: TileAccessPatternKind,
        /// Exclusive upper bound on `i`.
        loop_bound: u64,
    },
    /// Loaded element or arithmetic value that does not affect indexing.
    DataValue,
}

/// A Pliron SSA value paired with its generated Lean identifier and [`SsaKind`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypedSsa {
    /// Deterministic Lean identifier (`"arg0"`, `"v0"`, `"v1"`, ...).
    pub lean_name: String,
    /// Semantic type and provenance of the SSA value.
    pub kind: SsaKind,
}

/// Typed SSA lookup table mapping Pliron [`Value`]s to [`TypedSsa`] bindings.
#[derive(Clone, Debug, Default)]
pub struct SsaEnv {
    bindings: HashMap<Value, TypedSsa>,
    next_var_id: usize,
}

impl SsaEnv {
    /// Create an empty SSA environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of recorded SSA bindings.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Returns true if no SSA bindings have been recorded.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Bind an entry block argument `arg{index}` with its [`SsaKind`].
    pub fn bind_arg(&mut self, value: Value, index: usize, kind: SsaKind) -> String {
        let lean_name = format!("arg{index}");
        self.bindings.insert(
            value,
            TypedSsa {
                lean_name: lean_name.clone(),
                kind,
            },
        );
        lean_name
    }

    /// Bind an operation result to a fresh `v{N}` Lean name (or reuse its
    /// existing name during fixed-point refinement) and update its [`SsaKind`].
    pub fn bind_result(&mut self, value: Value, kind: SsaKind) -> String {
        if let Some(existing) = self.bindings.get_mut(&value) {
            existing.kind = kind;
            return existing.lean_name.clone();
        }
        let id = self.next_var_id;
        self.next_var_id += 1;
        let lean_name = format!("v{id}");
        self.bindings.insert(
            value,
            TypedSsa {
                lean_name: lean_name.clone(),
                kind,
            },
        );
        lean_name
    }

    /// Look up a previously defined SSA operand, rejecting forward, undefined,
    /// or external SSA references fail-closed.
    pub fn get(&self, value: &Value, scope: &str) -> Result<&TypedSsa, ExportError> {
        self.bindings.get(value).ok_or_else(|| {
            reject(
                scope,
                "operand is not an earlier result or an entry argument",
            )
        })
    }

    /// Look up the [`SsaKind`] of `value` if it is already bound.
    pub fn get_kind(&self, value: &Value) -> Option<SsaKind> {
        self.bindings.get(value).map(|b| b.kind.clone())
    }
}
