/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Renders a [`TranslatedFunction`] into a `.lean` source file.
//!
//! ```text
//!   TranslatedFunction
//!     ├── Scalar ──► emits `CudaOxide.Mir.Scalar.Expr` + `evaluate`
//!     └── Kernel ──► emits `CudaOxide.Mir.Kernel.KernelSummary1D` +
//!                    `NoOobIdx` / `DisjointWrites` / `WritesAllElems` theorems
//! ```

use crate::{
    ExportError, reject,
    spec::KernelVerificationSpec,
    ssa::{GlobalStripKind, TileAccessPatternKind},
    translate::{
        AccessKind, ExtractedAccess, GlobalStripAccess2D, PhasedSharedAccess, TranslatedFunction,
    },
};

/// Render any [`TranslatedFunction`] into a complete Lean 4 file.
pub fn render_lean_module(translated: &TranslatedFunction) -> Result<String, ExportError> {
    match translated {
        TranslatedFunction::Scalar {
            width,
            arity,
            let_lines,
            returned_var,
        } => Ok(format!(
            "-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n\
             -- SPDX-License-Identifier: Apache-2.0\n\
             -- Generated from scalar MIR. This file is a model, not a proof certificate.\n\
             import CudaOxide.Mir.Scalar\n\n\
             namespace CudaOxide.Generated\n\n\
             def program : CudaOxide.Mir.Scalar.Expr {width} {arity} :=\n{}\n  {returned_var}\n\n\
             def evaluate (inputs : Fin {arity} → BitVec {width}) : BitVec {width} := program.eval inputs\n\n\
             end CudaOxide.Generated\n",
            let_lines.join("\n"),
        )),
        TranslatedFunction::Kernel {
            arity,
            spec,
            accesses,
            ..
        } => render_verified_kernel_module(*arity, spec, accesses),
        TranslatedFunction::TiledKernel2D {
            spec,
            num_barriers,
            barriers_uniform,
            global_strip_accesses,
            shared_accesses,
            ..
        } => render_verified_tiled_2d_module(
            spec,
            *num_barriers,
            *barriers_uniform,
            global_strip_accesses,
            shared_accesses,
        ),
    }
}

/// Return the list of human-readable obligation descriptions proven for `translated`.
pub fn proved_obligation_descriptions(
    spec: &KernelVerificationSpec,
    global_strip_accesses: &[GlobalStripAccess2D],
) -> Vec<String> {
    let mut items = Vec::new();
    for &p in &spec.no_oob_idx {
        if spec.domain == 2 {
            let contract = match global_strip_accesses.iter().find(|a| a.param == p) {
                Some(GlobalStripAccess2D {
                    kind: GlobalStripKind::ColStrip,
                    ..
                }) => "CudaOxide.Contracts.NoOobColStrip",
                _ => "CudaOxide.Contracts.NoOobRowStrip",
            };
            items.push(format!("no_oob_idx(param {p}) -> {contract}"));
        } else {
            items.push(format!(
                "no_oob_idx(param {p}) -> CudaOxide.Contracts.NoOobIdx"
            ));
        }
    }
    for &p in &spec.disjoint_writes {
        items.push(format!(
            "disjoint_writes(param {p}) -> CudaOxide.Contracts.DisjointWrites"
        ));
    }
    for &p in &spec.writes_all_elems {
        items.push(format!(
            "writes_all_elems(param {p}) -> CudaOxide.Contracts.WritesAllElems"
        ));
    }
    for &s in &spec.no_oob_shared {
        items.push(format!(
            "no_oob_idx(shared {s}) -> CudaOxide.Contracts.NoOobTileAccess"
        ));
    }
    for &s in &spec.no_data_race {
        items.push(format!(
            "no_data_race(shared {s}) -> CudaOxide.Contracts.NoDataRace"
        ));
    }
    if spec.uniform_barriers {
        items.push("uniform_barriers -> CudaOxide.Contracts.UniformBarriers".to_string());
    }
    items
}

fn render_verified_kernel_module(
    arity: usize,
    spec: &KernelVerificationSpec,
    accesses: &[ExtractedAccess],
) -> Result<String, ExportError> {
    let block_size = spec.block_size;
    let mut access_entries = Vec::new();
    for access in accesses {
        let kind = match access.kind {
            AccessKind::Read => ".read",
            AccessKind::Write => ".write",
        };
        access_entries.push(format!(
            "    {{ kind := {kind}, targetParam := ⟨{}, by decide⟩, guardParam := ⟨{}, by decide⟩, indexOffset := {} }}",
            access.target_param, access.guard_param, access.index_offset
        ));
    }

    let mut theorems = Vec::new();
    let mut axiom_prints = Vec::new();

    for &param in &spec.no_oob_idx {
        let matching: Vec<_> = accesses
            .iter()
            .filter(|a| a.target_param == param)
            .collect();
        if matching.is_empty() {
            return Err(reject(
                "no_oob_idx",
                format!(
                    "parameter {param} is listed in `no_oob_idx` but never accessed in the kernel"
                ),
            ));
        }
        for (idx, access) in matching.iter().enumerate() {
            let thm_name = if matching.len() == 1 {
                format!("verify_no_oob_idx_param{param}")
            } else {
                format!("verify_no_oob_idx_param{param}_{idx}")
            };
            let guard = access.guard_param;
            let offset = access.index_offset;
            if offset == 0 && param == guard {
                theorems.push(format!(
                    "theorem {thm_name} (lengths : Fin {arity} → Nat) :\n    \
                     CudaOxide.Contracts.NoOobIdx {block_size} 0 (lengths ⟨{guard}, by decide⟩) (lengths ⟨{param}, by decide⟩) := by\n  \
                     exact CudaOxide.Proofs.Bounds.no_oob_idx_of_same_len"
                ));
            } else if offset == 0 && spec.requires_len_ge.contains(&(param, guard)) {
                theorems.push(format!(
                    "theorem {thm_name} (lengths : Fin {arity} → Nat)\n    \
                     (h_req : lengths ⟨{guard}, by decide⟩ ≤ lengths ⟨{param}, by decide⟩) :\n    \
                     CudaOxide.Contracts.NoOobIdx {block_size} 0 (lengths ⟨{guard}, by decide⟩) (lengths ⟨{param}, by decide⟩) := by\n  \
                     exact CudaOxide.Proofs.Bounds.no_oob_idx_of_guard_le h_req"
                ));
            } else {
                let req_binder = if spec.requires_len_ge.contains(&(param, guard)) {
                    format!(
                        "\n    (h_req : lengths ⟨{guard}, by decide⟩ ≤ lengths ⟨{param}, by decide⟩)"
                    )
                } else {
                    String::new()
                };
                theorems.push(format!(
                    "theorem {thm_name} (lengths : Fin {arity} → Nat){req_binder} :\n    \
                     CudaOxide.Contracts.NoOobIdx {block_size} {offset} (lengths ⟨{guard}, by decide⟩) (lengths ⟨{param}, by decide⟩) := by\n  \
                     intro block lane active\n  \
                     omega"
                ));
            }
            axiom_prints.push(format!("#print axioms {thm_name}"));
        }
    }

    for &param in &spec.disjoint_writes {
        let writes: Vec<_> = accesses
            .iter()
            .filter(|a| a.kind == AccessKind::Write && a.target_param == param)
            .collect();
        if writes.is_empty() {
            return Err(reject(
                "disjoint_writes",
                format!(
                    "parameter {param} is listed in `disjoint_writes` but never written by the kernel"
                ),
            ));
        }
        for (idx, write) in writes.iter().enumerate() {
            let thm_name = if writes.len() == 1 {
                format!("verify_disjoint_writes_param{param}")
            } else {
                format!("verify_disjoint_writes_param{param}_{idx}")
            };
            let offset = write.index_offset;
            theorems.push(format!(
                "theorem {thm_name} :\n    \
                 CudaOxide.Contracts.DisjointWrites {block_size} {offset} := by\n  \
                 exact CudaOxide.Proofs.Launch.disjoint_writes_1d"
            ));
            axiom_prints.push(format!("#print axioms {thm_name}"));
        }
    }

    for &param in &spec.writes_all_elems {
        let writes: Vec<_> = accesses
            .iter()
            .filter(|a| a.kind == AccessKind::Write && a.target_param == param)
            .collect();
        if writes.is_empty() {
            return Err(reject(
                "writes_all_elems",
                format!(
                    "parameter {param} is listed in `writes_all_elems` but never written by the kernel"
                ),
            ));
        }
        let write = writes[0];
        let thm_name = format!("verify_writes_all_elems_param{param}");
        if write.index_offset == 0 && write.guard_param == param {
            theorems.push(format!(
                "theorem {thm_name} (blocks : Nat) (lengths : Fin {arity} → Nat)\n    \
                 (hcoverage : lengths ⟨{param}, by decide⟩ ≤ blocks * {block_size}) :\n    \
                 CudaOxide.Contracts.WritesAllElems blocks {block_size} (lengths ⟨{param}, by decide⟩) := by\n  \
                 exact CudaOxide.Proofs.Launch.writes_all_elems_1d (by decide) hcoverage"
            ));
        } else {
            theorems.push(format!(
                "theorem {thm_name} (blocks : Nat) (lengths : Fin {arity} → Nat)\n    \
                 (hcoverage : lengths ⟨{param}, by decide⟩ ≤ blocks * {block_size}) :\n    \
                 CudaOxide.Contracts.WritesAllElems blocks {block_size} (lengths ⟨{param}, by decide⟩) := by\n  \
                 intro i hi\n  \
                 omega"
            ));
        }
        axiom_prints.push(format!("#print axioms {thm_name}"));
    }

    Ok(format!(
        "-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n\
         -- SPDX-License-Identifier: Apache-2.0\n\
         -- Generated from PostMem2Reg dialect-mir.\n\
         import CudaOxide\n\n\
         namespace CudaOxide.Generated\n\n\
         def kernelSummary : CudaOxide.Mir.Kernel.KernelSummary1D {arity} :=\n  \
         {{ blockSize := {block_size}\n  \
         , accesses := [\n{}\n    ]\n  }}\n\n\
         {}\n\n\
         {}\n\n\
         end CudaOxide.Generated\n",
        access_entries.join(",\n"),
        theorems.join("\n\n"),
        axiom_prints.join("\n"),
    ))
}

fn format_tile_pattern(pat: TileAccessPatternKind) -> String {
    match pat {
        TileAccessPatternKind::ThreadSlot { stride } => format!(".threadSlot {stride}"),
        TileAccessPatternKind::RowScan { stride } => format!(".rowScan {stride}"),
        TileAccessPatternKind::ColScan { stride } => format!(".colScan {stride}"),
    }
}

fn format_phased_access(a: &PhasedSharedAccess) -> String {
    let kind = match a.kind {
        AccessKind::Read => ".read",
        AccessKind::Write => ".write",
    };
    let pat = format_tile_pattern(a.pattern);
    format!(
        "{{ kind := {kind}, space := .shared, bufferId := {}, phase := {}, pattern := {pat}, loopBound := {} }}",
        a.shared_id, a.phase, a.loop_bound
    )
}

fn render_verified_tiled_2d_module(
    spec: &KernelVerificationSpec,
    num_barriers: usize,
    barriers_uniform: bool,
    global_strip_accesses: &[GlobalStripAccess2D],
    shared_accesses: &[PhasedSharedAccess],
) -> Result<String, ExportError> {
    let bx = u64::from(spec.block_size);
    let by = u64::from(spec.block_y);
    let access_entries: Vec<String> = shared_accesses
        .iter()
        .map(|a| format!("    {}", format_phased_access(a)))
        .collect();

    let mut theorems = Vec::new();
    let mut axiom_prints = Vec::new();

    for &param in &spec.no_oob_idx {
        let matching: Vec<_> = global_strip_accesses
            .iter()
            .filter(|a| a.param == param)
            .collect();
        if matching.is_empty() {
            return Err(reject(
                "no_oob_idx",
                format!(
                    "parameter {param} is listed in `no_oob_idx` but never accessed in the kernel"
                ),
            ));
        }
        for (idx, acc) in matching.iter().enumerate() {
            let thm_name = if matching.len() == 1 {
                format!("verify_no_oob_idx_param{param}")
            } else {
                format!("verify_no_oob_idx_param{param}_{idx}")
            };
            let (contract, lemma) = match acc.kind {
                GlobalStripKind::RowStrip => (
                    "CudaOxide.Contracts.NoOobRowStrip",
                    "CudaOxide.Proofs.Bounds.no_oob_row_strip",
                ),
                GlobalStripKind::ColStrip => (
                    "CudaOxide.Contracts.NoOobColStrip",
                    "CudaOxide.Proofs.Bounds.no_oob_col_strip",
                ),
            };
            theorems.push(format!(
                "theorem {thm_name} (bufferLen : Nat) :\n    \
                 {contract} bufferLen := by\n  \
                 exact {lemma}"
            ));
            axiom_prints.push(format!("#print axioms {thm_name}"));
        }
    }

    for &shared_id in &spec.no_oob_shared {
        let matching: Vec<_> = shared_accesses
            .iter()
            .filter(|a| a.shared_id == shared_id)
            .collect();
        if matching.is_empty() {
            return Err(reject(
                "no_oob_idx",
                format!(
                    "shared array {shared_id} is listed in `no_oob_idx` but never accessed in the kernel"
                ),
            ));
        }
        for (idx, acc) in matching.iter().enumerate() {
            let thm_name = format!("verify_no_oob_idx_shared{shared_id}_{idx}");
            let pat_str = format_tile_pattern(acc.pattern);
            let buf_len = acc.buffer_len;
            let loop_bound = acc.loop_bound;
            let fits = by * bx <= buf_len;
            let proof_line = match acc.pattern {
                TileAccessPatternKind::ThreadSlot { stride } if stride == bx && fits => {
                    "exact CudaOxide.Proofs.Bounds.no_oob_tile_slot (by decide)".to_string()
                }
                TileAccessPatternKind::RowScan { stride }
                    if stride == bx && loop_bound == bx && fits =>
                {
                    "exact CudaOxide.Proofs.Bounds.no_oob_tile_row_scan (by decide)".to_string()
                }
                TileAccessPatternKind::ColScan { stride }
                    if stride == bx && loop_bound == by && fits =>
                {
                    "exact CudaOxide.Proofs.Bounds.no_oob_tile_col_scan (by decide)".to_string()
                }
                _ => {
                    "intro tx ty i htx hty hi\n  simp [CudaOxide.Mir.Kernel.TileAccessPattern.address, CudaOxide.Mir.globalIndex]\n  omega"
                        .to_string()
                }
            };
            theorems.push(format!(
                "theorem {thm_name} :\n    \
                 CudaOxide.Contracts.NoOobTileAccess {bx} {by} {loop_bound} ({pat_str}) {buf_len} := by\n  \
                 {proof_line}"
            ));
            axiom_prints.push(format!("#print axioms {thm_name}"));
        }
    }

    for &shared_id in &spec.no_data_race {
        let matching: Vec<_> = shared_accesses
            .iter()
            .filter(|a| a.shared_id == shared_id)
            .collect();
        if matching.is_empty() {
            return Err(reject(
                "no_data_race",
                format!(
                    "shared array {shared_id} is listed in `no_data_race` but never accessed in the kernel"
                ),
            ));
        }
        for (i, a1) in matching.iter().enumerate() {
            for (j, a2) in matching.iter().enumerate().skip(i) {
                if a1.kind == AccessKind::Read && a2.kind == AccessKind::Read {
                    continue;
                }
                let thm_name = format!("verify_no_data_race_shared{shared_id}_{i}_{j}");
                let a1_str = format_phased_access(a1);
                let a2_str = format_phased_access(a2);
                let proof_line = if a1.phase != a2.phase {
                    "exact CudaOxide.Proofs.Launch.no_data_race_diff_phase (by decide)".to_string()
                } else if a1.pattern == (TileAccessPatternKind::ThreadSlot { stride: bx })
                    && a2.pattern == (TileAccessPatternKind::ThreadSlot { stride: bx })
                    && a1.loop_bound == a2.loop_bound
                {
                    "exact CudaOxide.Proofs.Launch.no_data_race_thread_slot".to_string()
                } else {
                    "intro tx₁ ty₁ i₁ tx₂ ty₂ i₂ htx₁ hty₁ hi₁ htx₂ hty₂ hi₂ _ _ _ hsame\n  \
                     simp [CudaOxide.Mir.Kernel.TileAccessPattern.address, CudaOxide.Mir.globalIndex] at hsame\n  \
                     omega"
                        .to_string()
                };
                theorems.push(format!(
                    "theorem {thm_name} :\n    \
                     CudaOxide.Contracts.NoDataRace {bx} {by}\n      {a1_str}\n      {a2_str} := by\n  \
                     {proof_line}"
                ));
                axiom_prints.push(format!("#print axioms {thm_name}"));
            }
        }
    }

    if spec.uniform_barriers {
        let thm_name = "verify_uniform_barriers";
        let proof_line = if barriers_uniform && num_barriers > 0 {
            "exact CudaOxide.Proofs.Launch.uniform_barriers_of_checked (by decide)"
        } else {
            "unfold CudaOxide.Contracts.UniformBarriers\n  omega"
        };
        theorems.push(format!(
            "theorem {thm_name} :\n    \
             CudaOxide.Contracts.UniformBarriers {barriers_uniform} {num_barriers} := by\n  \
             {proof_line}"
        ));
        axiom_prints.push(format!("#print axioms {thm_name}"));
    }

    Ok(format!(
        "-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n\
         -- SPDX-License-Identifier: Apache-2.0\n\
         -- Generated from PostMem2Reg dialect-mir (2D Phased Shared-Memory Kernel).\n\
         import CudaOxide\n\n\
         namespace CudaOxide.Generated\n\n\
         def kernelSummary : CudaOxide.Mir.Kernel.TiledKernelSummary2D :=\n  \
         {{ blockDimX := {bx}\n  \
         , blockDimY := {by}\n  \
         , numBarriers := {num_barriers}\n  \
         , barriersUniform := {barriers_uniform}\n  \
         , sharedAccesses := [\n{}\n    ]\n  }}\n\n\
         {}\n\n\
         {}\n\n\
         end CudaOxide.Generated\n",
        access_entries.join(",\n"),
        theorems.join("\n\n"),
        axiom_prints.join("\n"),
    ))
}
