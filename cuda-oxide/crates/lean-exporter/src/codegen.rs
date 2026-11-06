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
    translate::{AccessKind, ExtractedAccess, TranslatedFunction},
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
    }
}

/// Return the list of human-readable obligation descriptions proven for `spec`.
pub fn proved_obligation_descriptions(spec: &KernelVerificationSpec) -> Vec<String> {
    let mut items = Vec::new();
    for &p in &spec.no_oob_idx {
        items.push(format!("no_oob_idx(param {p}) -> CudaOxide.Contracts.NoOobIdx"));
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
