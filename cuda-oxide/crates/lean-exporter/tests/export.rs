/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use combine::Parser;
use lean_exporter::{export_function, export_named_source};
use pliron::{
    builtin::{
        attributes::TypeAttr,
        types::{IntegerType, Signedness},
    },
    context::{Context, Ptr},
    identifier::Identifier,
    irfmt::parsers::spaced,
    linked_list::ContainsLinkedList,
    operation::Operation,
    parsable::parse_from_str,
};

const FIXTURE: &str = include_str!("../fixtures/arithmetic.mir");
const VECADD_FIXTURE: &str = include_str!("../fixtures/vecadd.mir");
const SGEMM_TILED_FIXTURE: &str = include_str!("../fixtures/sgemm_tiled.mir");

fn export(source: &str) -> String {
    export_named_source(source, "global_index").expect("valid scalar MIR")
}

fn reject(source: &str, expected: &str) {
    let error = export_named_source(source, "global_index").expect_err("must reject input");
    assert!(
        error.to_string().contains(expected),
        "unexpected error: {error}"
    );
}

fn constant(width: u32, literal: &str) -> String {
    format!(
        "mir.func @constant: builtin.function <() -> (builtin.integer ui{width})> [] {{
           ^entry():
             c = mir.constant () [] [value: builtin.integer <{literal}: ui{width}>]: <() -> (builtin.integer ui{width})>;
             mir.return (c) [] []: <(builtin.integer ui{width}) -> ()>
         }}"
    )
}

fn parsed() -> (Context, Ptr<Operation>) {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let source = FIXTURE
        .lines()
        .filter(|line| !line.starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    let function = parse_from_str(
        spaced(Operation::top_level_parser()).skip(combine::eof()),
        &mut ctx,
        &source,
    )
    .expect("parse fixture");
    (ctx, function)
}

#[test]
fn exports_actual_ssa_and_preserves_operand_order() {
    let baseline = export(FIXTURE);
    assert!(baseline.contains("def program : CudaOxide.Mir.Scalar.Expr 64 3"));
    assert!(baseline.contains(".mul arg0 arg1"));
    assert!(baseline.contains(".add v0 arg2"));
    assert!(
        baseline.contains(
            "def evaluate (inputs : Fin 3 → BitVec 64) : BitVec 64 := program.eval inputs"
        )
    );

    let operator_mutation = export(&FIXTURE.replace("mir.mul", "mir.sub"));
    assert_ne!(baseline, operator_mutation);
    assert!(operator_mutation.contains(".sub arg0 arg1"));

    let operand_mutation = export(&FIXTURE.replace("(product, lane)", "(lane, product)"));
    assert_ne!(baseline, operand_mutation);
    assert!(operand_mutation.contains(".add arg2 v0"));

    let return_mutation = export(&FIXTURE.replace("mir.return (index)", "mir.return (block)"));
    assert_ne!(baseline, return_mutation);
    assert!(return_mutation.contains("\n  arg0\n"));
}

#[test]
fn supports_all_four_unsigned_widths() {
    for width in [8, 16, 32, 64] {
        let output = export(&FIXTURE.replace("ui64", &format!("ui{width}")));
        assert!(output.contains(&format!("Expr {width} 3")));
    }
}

#[test]
fn preserves_high_bit_unsigned_constants_and_zero_arity() {
    for (width, value) in [
        (8, "255"),
        (16, "65535"),
        (32, "4294967295"),
        (64, "18446744073709551615"),
    ] {
        let output = export_named_source(&constant(width, value), "constant").unwrap();
        assert!(output.contains(&format!(".literal (BitVec.ofNat {width} {value})")));
        assert!(output.contains(&format!("Expr {width} 0")));
    }
    let changed = export_named_source(&constant(8, "254"), "constant").unwrap();
    assert_ne!(
        changed,
        export_named_source(&constant(8, "255"), "constant").unwrap()
    );
    assert!(export_named_source(&constant(8, "256"), "constant").is_err());
}

#[test]
fn accepts_only_integer_builtin_constants() {
    let input = "mir.func @constant: builtin.function <() -> (builtin.integer ui8)> [] {
      ^entry():
        c = builtin.constant <builtin.integer <255: ui8>> : builtin.integer ui8;
        mir.return (c) [] []: <(builtin.integer ui8) -> ()>
    }";
    assert!(
        export_named_source(input, "constant")
            .unwrap()
            .contains("BitVec.ofNat 8 255")
    );
    let bad = input.replace("builtin.integer <255: ui8>", "builtin.string \"255\"");
    assert!(
        export_named_source(&bad, "constant")
            .unwrap_err()
            .to_string()
            .contains("malformed attribute")
    );
}

#[test]
fn names_do_not_become_lean_identifiers_or_source() {
    let renamed = FIXTURE
        .replace("global_index", "theorem")
        .replace("product", "namespace")
        .replace("lane", "axiom");
    assert_eq!(
        export(FIXTURE),
        export_named_source(&renamed, "theorem").unwrap()
    );
    assert_eq!(export(FIXTURE), export(FIXTURE));
}

#[test]
fn rejects_signed_signless_float_and_unsupported_width() {
    for ty in ["si64", "i64", "ui128", "ui1"] {
        reject(&FIXTURE.replace("ui64", ty), "unsigned integer of width");
    }
    reject(
        &FIXTURE.replace("builtin.integer ui64", "builtin.fp32"),
        "only unsigned integer types",
    );
    reject(
        &FIXTURE.replace(
            "builtin.integer ui64",
            "mir.ptr <builtin.integer ui64, mutable: false, addrspace: 0, kind: RawConst>",
        ),
        "only unsigned integer types",
    );
}

#[test]
fn rejects_unknown_memory_checked_and_dead_operations() {
    reject(&FIXTURE.replace("mir.mul", "mir.not_registered"), "parse");
    for unsupported in ["mir.checked_add", "mir.load", "mir.call", "mir.assert"] {
        reject(
            &FIXTURE.replace("mir.mul", unsupported),
            "unsupported operation",
        );
    }
    let with_dead = FIXTURE.replace(
        "    mir.return",
        "    ignored = mir.neg (block) [] []: <(builtin.integer ui64) -> (builtin.integer ui64)>;\n    mir.return",
    );
    reject(&with_dead, "unsupported operation");
}

#[test]
fn rejects_attrs_regions_successors_and_extra_blocks() {
    let attributes = FIXTURE.replacen("[] []:", "[] [unexpected: builtin.bool true]:", 1);
    reject(&attributes, "unsupported attribute `unexpected`");
    let block_attribute = FIXTURE.replace(
        "lane: builtin.integer ui64):",
        "lane: builtin.integer ui64) [unexpected: builtin.bool true]:",
    );
    reject(&block_attribute, "unsupported attribute");
    let successors = FIXTURE.replacen("[] []:", "[^entry] []:", 1);
    reject(&successors, "successors are unsupported");
    let with_region = FIXTURE.replacen(
        "-> (builtin.integer ui64)>;",
        "-> (builtin.integer ui64)> {};",
        1,
    );
    reject(&with_region, "regions and successors are unsupported");
    let extra_block = FIXTURE.replace(
        "\n}",
        "\n ^second(other: builtin.integer ui64):\n mir.return (other) [] []: <(builtin.integer ui64) -> ()>\n}",
    );
    reject(&extra_block, "exactly one nonempty basic block");
}

#[test]
fn rejects_forward_undefined_mixed_and_malformed_ssa() {
    reject(
        &FIXTURE.replace("(block, block_size)", "(index, block_size)"),
        "not an earlier result",
    );
    reject(
        &FIXTURE.replace("(block, block_size)", "(undefined, block_size)"),
        "parse",
    );
    reject(&FIXTURE.replace("(block, block_size)", "(block)"), "parse");
    let mixed = FIXTURE.replacen("builtin.integer ui64", "builtin.integer ui32", 1);
    reject(&mixed, "all arguments must have the same type");
    let no_return = FIXTURE
        .lines()
        .filter(|line| !line.contains("mir.return"))
        .collect::<Vec<_>>()
        .join("\n");
    reject(&no_return.replace(">;\n}", ">\n}"), "missing mir.return");
}

#[test]
fn api_rejects_missing_or_wrong_function_type_without_panicking() {
    let (ctx, function) = parsed();
    function
        .deref_mut(&ctx)
        .attributes
        .0
        .retain(|key, _| key.to_string() != "mir_func_type");
    assert!(
        export_function(&ctx, function)
            .unwrap_err()
            .to_string()
            .contains("malformed attribute")
    );

    let (ctx, function) = parsed();
    let integer = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    function.deref_mut(&ctx).attributes.set(
        Identifier::try_new("mir_func_type".into()).unwrap(),
        TypeAttr::new(integer.into()),
    );
    assert!(
        export_function(&ctx, function)
            .unwrap_err()
            .to_string()
            .contains("builtin.function type")
    );
}

#[test]
fn api_rejects_wrong_result_count_without_panicking() {
    let (ctx, function) = parsed();
    let block = function
        .deref(&ctx)
        .get_region(0)
        .deref(&ctx)
        .get_head()
        .unwrap();
    let add = block.deref(&ctx).iter(&ctx).nth(1).unwrap();
    let integer = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    Operation::push_result(add, &ctx, integer.into());
    assert!(
        export_function(&ctx, function)
            .unwrap_err()
            .to_string()
            .contains("incorrect operand or result count")
    );
}

#[test]
fn selects_exact_function_and_rejects_trailing_input_and_duplicate_symbols() {
    assert!(
        export_named_source(FIXTURE, "missing")
            .unwrap_err()
            .to_string()
            .contains("was not found")
    );
    reject(&format!("{FIXTURE}\ntrailing"), "parse");
    let body = FIXTURE
        .lines()
        .filter(|line| !line.starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    let module = format!("builtin.module @test {{ ^entry(): {body} }}");
    assert_eq!(export(&module), export(FIXTURE));
    let duplicate = format!("builtin.module @test {{ ^entry(): {body}; {body} }}");
    reject(&duplicate, "duplicate function symbol");
}

#[test]
fn cli_reports_errors_only_on_stderr_and_returns_failure() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_lean-exporter"))
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/arithmetic.mir"
        ))
        .arg("missing")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("was not found"));
}

#[test]
fn cli_inspect_shows_the_exact_input_and_export_only_after_success() {
    let run = |function: &str| {
        std::process::Command::new(env!("CARGO_BIN_EXE_lean-exporter"))
            .arg("--inspect")
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/fixtures/arithmetic.mir"
            ))
            .arg(function)
            .output()
            .unwrap()
    };
    let output = run("global_index");
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(
        text,
        format!(
            "=== MIR input ===\n{FIXTURE}\n=== Generated Lean ===\n{}",
            export(FIXTURE)
        )
    );

    let failed = run("missing");
    assert!(!failed.status.success());
    assert!(failed.stdout.is_empty());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("was not found"));
}

#[test]
fn exports_verified_vecadd_kernel_and_rejects_unguarded_accesses() {
    let lean = export_named_source(VECADD_FIXTURE, "vecadd").expect("valid verified vecadd MIR");
    assert!(lean.contains("def kernelSummary : CudaOxide.Mir.Kernel.KernelSummary1D 3"));
    assert!(lean.contains("theorem verify_no_oob_idx_param0"));
    assert!(lean.contains("theorem verify_no_oob_idx_param1"));
    assert!(lean.contains("theorem verify_no_oob_idx_param2"));
    assert!(lean.contains("theorem verify_disjoint_writes_param2"));
    assert!(lean.contains("theorem verify_writes_all_elems_param2"));

    // Scenario 4a: Moving a load into the unguarded exit block must be rejected fail-closed.
    let unguarded_read = VECADD_FIXTURE.replace(
        "^exit():\n    mir.return",
        "^exit():\n    bad_ptr = mir.extract_field (a) [] [index: mir.field_index 0]: <(mir.slice <builtin.fp32 ,mutable:false,kind:SharedRef>) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;\n    bad_elem = mir.ptr_offset (bad_ptr, idx) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>, builtin.integer ui64) -> (mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>)>;\n    bad_val = mir.load (bad_elem) [] []: <(mir.ptr <builtin.fp32 ,mutable:false,addrspace:0,kind:Erased>) -> (builtin.fp32)>;\n    mir.return",
    );
    let err = export_named_source(&unguarded_read, "vecadd").unwrap_err();
    assert!(err.to_string().contains("unguarded read"));

    // Scenario 4b: Moving a store into the unguarded exit block must be rejected fail-closed.
    let unguarded_write = VECADD_FIXTURE
        .replace("builtin.fp32", "builtin.integer ui64")
        .replace(
            "^exit():\n    mir.return",
            "^exit():\n    bad_c_ptr = mir.extract_field (c) [] [index: mir.field_index 0]: <(mir.disjoint_slice <builtin.integer ui64 ,[]>) -> (mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:0,kind:RawMut>)>;\n    bad_c_elem = mir.ptr_offset (bad_c_ptr, idx) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:0,kind:RawMut>, builtin.integer ui64) -> (mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:0,kind:RawMut>)>;\n    mir.store (bad_c_elem, idx) [] []: <(mir.ptr <builtin.integer ui64 ,mutable:true,addrspace:0,kind:RawMut>, builtin.integer ui64) -> ()>;\n    mir.return",
        );
    let err = export_named_source(&unguarded_write, "vecadd").unwrap_err();
    assert!(
        err.to_string().contains("unguarded write"),
        "unexpected error for unguarded write: {err}"
    );

    // Scenario 3: Inclusive branch guard (`idx <= c.len()` via `mir.le`) instead of `<` (`mir.lt`).
    let inclusive_guard = VECADD_FIXTURE.replace("mir.lt (idx, c_len)", "mir.le (idx, c_len)");
    let err = export_named_source(&inclusive_guard, "vecadd").unwrap_err();
    assert!(
        err.to_string().contains("unguarded read")
            || err.to_string().contains("unsupported operation `mir.le`"),
        "unexpected error for inclusive guard: {err}"
    );

    // Scenario 5a: Claiming `disjoint_writes` on read-only parameter 0 (`a`).
    let bad_disjoint = VECADD_FIXTURE.replace(
        "verify_disjoint_writes: builtin.string \"2\"",
        "verify_disjoint_writes: builtin.string \"0\"",
    );
    let err = export_named_source(&bad_disjoint, "vecadd").unwrap_err();
    assert!(
        err.to_string()
            .contains("parameter 0 is read-only and cannot satisfy a write obligation"),
        "unexpected error: {err}"
    );

    // Scenario 5b: Claiming `no_oob_idx` / `disjoint_writes` on parameter 2 (`c`) when `c` is never written.
    let unwritten_c = VECADD_FIXTURE
        .lines()
        .filter(|line| !line.contains("mir.store (c_elem, sum)"))
        .collect::<Vec<_>>()
        .join("\n");
    let err = export_named_source(&unwritten_c, "vecadd").unwrap_err();
    assert!(
        err.to_string()
            .contains("parameter 2 is listed in `no_oob_idx` but never accessed"),
        "unexpected error: {err}"
    );
    let unwritten_c_disjoint = unwritten_c.replace(
        "verify_no_oob_idx: builtin.string \"0,1,2\"",
        "verify_no_oob_idx: builtin.string \"0,1\"",
    );
    let err = export_named_source(&unwritten_c_disjoint, "vecadd").unwrap_err();
    assert!(
        err.to_string()
            .contains("parameter 2 is listed in `disjoint_writes` but never written"),
        "unexpected error: {err}"
    );

    // Scenario 5c: Indexing a slice by a non-thread-index value (`c_len` instead of `idx`).
    let bad_index_expr = VECADD_FIXTURE.replace(
        "a_elem = mir.ptr_offset (a_ptr, idx)",
        "a_elem = mir.ptr_offset (a_ptr, c_len)",
    );
    let err = export_named_source(&bad_index_expr, "vecadd").unwrap_err();
    assert!(
        err.to_string()
            .contains("pointer offset must index a kernel slice parameter by the 1D thread index"),
        "unexpected error: {err}"
    );
}

#[test]
fn lean_typechecker_accepts_valid_vecadd_and_rejects_false_kernel_claims() {
    let lean_dir = lean_exporter::runner::default_lean_package_dir();
    let baseline = export_named_source(VECADD_FIXTURE, "vecadd").expect("valid vecadd MIR");

    // If `lake` is not installed in a minimal container environment, skip the live
    // Lean subprocess assertions (covered by the dedicated `lean-verification` CI job).
    if let Err(err) = lean_exporter::runner::check_generated_lean(&baseline, &lean_dir) {
        if err.to_string().contains("could not locate `lake`") {
            return;
        }
        panic!("valid vecadd MIR must pass Lean 4 verification, got: {err}");
    }

    // Scenario 1: Missing `a.len() >= c.len()` (`0>=2` dropped from `contract_requires_len_ge`).
    // Stage 1 exports the theorem without `h_req`, and Stage 2 (Lean `omega`) rejects it.
    let missing_requires = VECADD_FIXTURE.replace("0>=2,1>=2", "1>=2");
    let missing_requires_lean =
        export_named_source(&missing_requires, "vecadd").expect("exports theorem obligation");
    let err = lean_exporter::runner::check_generated_lean(&missing_requires_lean, &lean_dir)
        .expect_err("Lean must reject `no_oob_idx(a)` when `a.len() >= c.len()` is missing");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for missing requires: {err}"
    );

    // Scenario 2: Off-by-one neighbor access `a[idx + 1]` under `if idx < c.len()`.
    // Stage 1 extracts `indexOffset := 1`, and Stage 2 (Lean `omega`) rejects it because
    // when `a.len() == c.len()` and `idx == c.len() - 1`, `idx + 1 == a.len()` is OOB!
    let off_by_one = VECADD_FIXTURE.replace(
        "a_elem = mir.ptr_offset (a_ptr, idx)",
        "one = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;\n    idx_plus_one = mir.add (idx, one) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n    a_elem = mir.ptr_offset (a_ptr, idx_plus_one)",
    );
    let off_by_one_lean =
        export_named_source(&off_by_one, "vecadd").expect("exports offset=1 obligation");
    assert!(off_by_one_lean.contains("indexOffset := 1"));
    assert!(off_by_one_lean.contains("CudaOxide.Contracts.NoOobIdx 256 1"));
    let err = lean_exporter::runner::check_generated_lean(&off_by_one_lean, &lean_dir)
        .expect_err("Lean must reject off-by-one `a[idx + 1]` access");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for off-by-one access: {err}"
    );

    // Scenario 6: Admitted proof (`sorry`) must be caught by the `#print axioms` gate.
    let spoofed_sorry = baseline.replace(
        "exact CudaOxide.Proofs.Bounds.no_oob_idx_of_guard_le h_req",
        "sorry",
    );
    let err = lean_exporter::runner::check_generated_lean(&spoofed_sorry, &lean_dir)
        .expect_err("Lean runner must reject `sorry` proofs");
    assert!(
        err.to_string().contains("sorry"),
        "unexpected error for sorry proof: {err}"
    );
}

#[test]
fn verifies_2d_sgemm_tiled_views_and_rejects_shared_races_oob_and_divergent_barriers() {
    let lean_dir = lean_exporter::runner::default_lean_package_dir();
    let baseline = export_named_source(SGEMM_TILED_FIXTURE, "sgemm_tiled_views")
        .expect("valid sgemm_tiled_views MIR");
    assert!(baseline.contains("def kernelSummary : CudaOxide.Mir.Kernel.TiledKernelSummary2D"));
    assert!(baseline.contains("theorem verify_no_oob_idx_shared0_0"));
    assert!(baseline.contains("theorem verify_no_data_race_shared0_0_1"));
    assert!(baseline.contains("theorem verify_uniform_barriers"));

    // Stage 1 negative check: indexing SharedArray by a non-2D-tile expression (`k` instead of `smem_idx`).
    let bad_shared_offset = SGEMM_TILED_FIXTURE.replace(
        "slot_a = mir.ptr_offset (tile_a_mut, smem_idx)",
        "k_u64 = mir.cast (k) [] [cast_kind: mir.cast_kind IntToInt]: <(builtin.integer ui32) -> (builtin.integer ui64)>;\n    slot_a = mir.ptr_offset (tile_a_mut, k_u64)",
    );
    let err = export_named_source(&bad_shared_offset, "sgemm_tiled_views").unwrap_err();
    assert!(
        err.to_string()
            .contains("shared memory pointer offset must use a recognized 2D tile coordinate"),
        "unexpected error: {err}"
    );

    if let Err(err) = lean_exporter::runner::check_generated_lean(&baseline, &lean_dir) {
        if err.to_string().contains("could not locate `lake`") {
            return;
        }
        panic!("valid sgemm_tiled_views MIR must pass Lean 4 verification, got: {err}");
    }

    // Stage 2 Negative Scenario A: Missing first `sync_threads()` (`nvvm.barrier0`) between
    // Phase 0 writes (`TILE_A[ty * 16 + tx]`) and Phase 1 reads (`TILE_A[ty * 16 + i]`).
    // Both accesses land in Phase 0, and Lean 4 rejects `NoDataRace` with a counterexample!
    let missing_barrier =
        SGEMM_TILED_FIXTURE.replacen("    nvvm.barrier0 () [] []: <() -> ()>;\n", "", 1);
    let missing_barrier_lean = export_named_source(&missing_barrier, "sgemm_tiled_views")
        .expect("exports phase-collapsed obligation");
    let err = lean_exporter::runner::check_generated_lean(&missing_barrier_lean, &lean_dir)
        .expect_err("Lean must reject missing `sync_threads()` shared-memory data race");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for missing sync_threads: {err}"
    );

    // Stage 2 Negative Scenario B: Shrinking `TILE_A` from 256 to 128 elements under a 16x16 block.
    // Lean 4 rejects `NoOobTileAccess` because `ty * 16 + tx` can reach 255 >= 128.
    let small_tile = SGEMM_TILED_FIXTURE.replacen(
        "size: builtin.integer <256: i64>",
        "size: builtin.integer <128: i64>",
        1,
    );
    let small_tile_lean =
        export_named_source(&small_tile, "sgemm_tiled_views").expect("exports OOB tile obligation");
    let err = lean_exporter::runner::check_generated_lean(&small_tile_lean, &lean_dir)
        .expect_err("Lean must reject undersized SharedArray<f32, 128> for 16x16 block");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for undersized shared tile: {err}"
    );

    // Stage 2 Negative Scenario C: Non-uniform / divergent barrier (early conditional return
    // before the barrier loop). Lean 4 rejects `UniformBarriers false ...`.
    let divergent_barrier = SGEMM_TILED_FIXTURE.replace(
        "mir.goto (zero_f32, zero_u32) [^tile_header] []: <(builtin.fp32, builtin.integer ui32) -> ()>",
        "early_cond = mir.lt (tx, k) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer i1)>;\n    mir.cond_br (early_cond) [^exit, ^enter_loop] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>\n\n  ^enter_loop():\n    mir.goto (zero_f32, zero_u32) [^tile_header] []: <(builtin.fp32, builtin.integer ui32) -> ()>",
    );
    let divergent_lean = export_named_source(&divergent_barrier, "sgemm_tiled_views")
        .expect("exports UniformBarriers false obligation");
    assert!(divergent_lean.contains("barriersUniform := false"));
    let err = lean_exporter::runner::check_generated_lean(&divergent_lean, &lean_dir)
        .expect_err("Lean must reject divergent barrier when early return skips barrier loop");
    assert!(
        err.to_string().contains("omega could not prove the goal"),
        "unexpected Lean error for divergent barrier: {err}"
    );

    // Stage 2 (`sgemm_tiled_raw`): Verify 2D global slice `RowStrip` (`a`) and `ColStrip` (`b`)
    // are extracted from MIR pointer-offset math rather than parameter list order.
    let raw_baseline =
        export_named_source(SGEMM_TILED_FIXTURE, "sgemm_tiled_raw").expect("valid sgemm_tiled_raw");
    assert!(raw_baseline.contains(
        "theorem verify_no_oob_idx_param4 (bufferLen : Nat) :\n    CudaOxide.Contracts.NoOobRowStrip bufferLen"
    ));
    assert!(raw_baseline.contains(
        "theorem verify_no_oob_idx_param5 (bufferLen : Nat) :\n    CudaOxide.Contracts.NoOobColStrip bufferLen"
    ));
    lean_exporter::runner::check_generated_lean(&raw_baseline, &lean_dir)
        .expect("valid sgemm_tiled_raw MIR must pass Lean 4 verification");

    // Reversing the attribute list (`"5,4"` instead of `"4,5"`) must still map param 5 (`b`)
    // to `NoOobColStrip` and param 4 (`a`) to `NoOobRowStrip`.
    let swapped_order = SGEMM_TILED_FIXTURE.replace(
        "verify_no_oob_idx: builtin.string \"4,5\"",
        "verify_no_oob_idx: builtin.string \"5,4\"",
    );
    let swapped_lean =
        export_named_source(&swapped_order, "sgemm_tiled_raw").expect("swapped order still valid");
    assert!(swapped_lean.contains(
        "theorem verify_no_oob_idx_param5 (bufferLen : Nat) :\n    CudaOxide.Contracts.NoOobColStrip bufferLen"
    ));
    assert!(swapped_lean.contains(
        "theorem verify_no_oob_idx_param4 (bufferLen : Nat) :\n    CudaOxide.Contracts.NoOobRowStrip bufferLen"
    ));

    // Negative check: indexing global slice `a` by a non-strip index (`a_len` instead of `a_idx`)
    // must be rejected fail-closed by `translate.rs`.
    let bad_global_offset = SGEMM_TILED_FIXTURE.replace(
        "a_elem = mir.ptr_offset (a_ptr, a_idx)",
        "a_elem = mir.ptr_offset (a_ptr, a_len)",
    );
    let err = export_named_source(&bad_global_offset, "sgemm_tiled_raw").unwrap_err();
    assert!(
        err.to_string()
            .contains("2D global slice pointer offset must use a row-strip"),
        "unexpected error for bad 2D global slice offset: {err}"
    );
}
