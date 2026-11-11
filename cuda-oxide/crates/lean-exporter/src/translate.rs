/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Single unified `dialect-mir` translator powered by [`SsaEnv`].
//!
//! Both pure scalar functions (`arithmetic.mir`) and verified GPU kernels
//! (`vecadd.mir`) walk their MIR instructions through this module using the
//! same typed lookup table (`HashMap<Value, TypedSsa>`):
//!
//! ```text
//!                     translate_function(ctx, mir_func)
//!                                    │
//!                       Builds `SsaEnv` for entry args
//!                                    │
//!             ┌──────────────────────┴──────────────────────┐
//!             ▼                                             ▼
//!     Scalar `mir.func`                         Verified Kernel `mir.func`
//!   • Binds args as `ScalarWord`              • Binds args as `BufferParam(i)`
//!   • Walks ops, checks `ScalarWord`,         • Walks ops, tracks `Index1D`,
//!     emits `let vN := .mul/.add ...`           `SliceLen`, `IndexLtLen`, `ElementPtr`
//!   • Returns `TranslatedFunction::Scalar`    • Returns `TranslatedFunction::Kernel`
//! ```

use std::collections::{BTreeSet, HashMap};

use pliron::{
    attribute::AttributeDict,
    basic_block::BasicBlock,
    builtin::{
        attributes::{IdentifierAttr, IntegerAttr, TypeAttr},
        type_interfaces::FunctionTypeInterface,
        types::{FunctionType, IntegerType},
    },
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::{Operation, verify_operation},
    r#type::{TypeHandle, Typed},
};

use crate::{
    ExportError, check_attributes, get_attribute, reject,
    spec::{
        KernelVerificationSpec, get_string_attr, has_verification_attributes,
        parse_kernel_verification_spec,
    },
    ssa::{GlobalStripKind, GuardExpr, IdxExpr, SsaEnv, SsaKind, TileAccessPatternKind},
};

/// Direction of an extracted buffer access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AccessKind {
    /// Read from a slice or buffer parameter (`mir.load`).
    Read,
    /// Write to a mutable slice or `DisjointSlice` (`mir.store`).
    Write,
}

/// One guarded 1D buffer access translated from `PostMem2Reg` `dialect-mir`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExtractedAccess {
    /// Read or write operation.
    pub kind: AccessKind,
    /// Accessed kernel parameter index.
    pub target_param: usize,
    /// Kernel parameter index whose `.len()` bounds the dominating branch guard.
    pub guard_param: usize,
    /// Constant element offset added to the 1D thread index (`0` for `buf[idx]`).
    pub index_offset: u64,
}

/// One phased shared-memory tile or reduction access extracted from `PostMem2Reg` `dialect-mir`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PhasedSharedAccess {
    /// Read or write operation.
    pub kind: AccessKind,
    /// Static `SharedArray` index (`0` for `TILE_A`, `1` for `TILE_B`, ...).
    pub shared_id: usize,
    /// Short name of the `SharedArray` static (`"TILE_A"`, `"WARP_SUMS"`, ...).
    pub shared_name: String,
    /// Declared element count of the `SharedArray`.
    pub buffer_len: u64,
    /// Barrier phase index (`0` before first `sync_threads()`, `1` after first, ...).
    pub phase: usize,
    /// Coordinate formula used to index the shared buffer.
    pub pattern: TileAccessPatternKind,
    /// Exclusive upper bound on loop index `i` (`1` outside a scan loop).
    pub loop_bound: u64,
    /// Whether this access is dominated by a leader guard (`lane == 0` / `tx % W == 0`).
    pub leader_only: bool,
    /// Dominating upper bound on `tx` (`block_size` when unguarded, or `N` under `if tid < N`).
    pub active_threads: u64,
    /// Optional residual branch guard (`GuardExpr::True` when unguarded or covered by `leader_only` / `active_threads`).
    pub guard: GuardExpr,
}

/// One 2D global slice strip access (`start + col` or `start + row * stride`)
/// extracted from `PostMem2Reg` `dialect-mir`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GlobalStripAccess2D {
    /// Accessed kernel slice parameter index (`0..arity`).
    pub param: usize,
    /// Row-strip (`start + col`) vs. column-strip (`start + row * stride`) shape.
    pub kind: GlobalStripKind,
}

/// Output of the unified MIR translator before rendering to `.lean` text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranslatedFunction {
    /// Pure single-block scalar function (`CudaOxide.Mir.Scalar.Expr`).
    Scalar {
        /// Bit width (`8`, `16`, `32`, or `64`).
        width: u32,
        /// Number of entry arguments.
        arity: usize,
        /// Ordered `let argN / vN := ...` lines.
        let_lines: Vec<String>,
        /// Final returned variable name (`"arg0"`, `"v1"`, ...).
        returned_var: String,
    },
    /// Verified 1D GPU kernel (`CudaOxide.Mir.Kernel.KernelSummary1D` + contracts).
    Kernel {
        /// Number of kernel parameters.
        arity: usize,
        /// Parsed `#[launch_contract]` and `#[verify(...)]` specification.
        spec: KernelVerificationSpec,
        /// Guarded buffer reads and writes extracted from the kernel CFG.
        accesses: Vec<ExtractedAccess>,
        /// `mir.assert` bounds-check operations proven redundant by `no_oob_idx`.
        proven_assert_ops: Vec<Ptr<Operation>>,
    },
    /// Verified 2D phased shared-memory GPU kernel (`TiledKernelSummary2D` + contracts).
    TiledKernel2D {
        /// Parsed `#[launch_contract]` and `#[verify(...)]` specification.
        spec: KernelVerificationSpec,
        /// Number of `sync_threads()` barriers in the tile loop.
        num_barriers: usize,
        /// Whether all barriers execute in thread-uniform control flow.
        barriers_uniform: bool,
        /// 2D global slice strip accesses extracted from the kernel CFG.
        global_strip_accesses: Vec<GlobalStripAccess2D>,
        /// Phased shared-memory reads and writes extracted from the kernel CFG.
        shared_accesses: Vec<PhasedSharedAccess>,
        /// `mir.assert` bounds-check operations proven redundant by `no_oob_idx`.
        proven_assert_ops: Vec<Ptr<Operation>>,
    },
}

/// Translate any supported `mir.func` (scalar function or verified GPU kernel)
/// using [`SsaEnv`].
pub fn translate_function(
    ctx: &Context,
    function: Ptr<Operation>,
) -> Result<TranslatedFunction, ExportError> {
    if has_verification_attributes(ctx, function) {
        translate_verified_kernel(ctx, function)
    } else {
        translate_scalar_function(ctx, function)
    }
}

/// Validate that `ty` is an unsigned integer type of width `8`, `16`, `32`, or `64`
/// and return its bit width.
fn unsigned_width(ctx: &Context, ty: TypeHandle) -> Result<u32, ExportError> {
    let type_ref = ty.deref(ctx);
    let int = type_ref.downcast_ref::<IntegerType>().ok_or_else(|| {
        reject(
            "function signature",
            "only unsigned integer types are supported",
        )
    })?;
    if !int.is_unsigned() || !matches!(int.width(), 8 | 16 | 32 | 64) {
        return Err(reject(
            "function signature",
            "expected an unsigned integer of width 8, 16, 32, or 64",
        ));
    }
    Ok(int.width())
}

/// Translate a single-block pure scalar integer `mir.func` into a
/// [`TranslatedFunction::Scalar`] `BitVec` AST using [`SsaEnv`].
fn translate_scalar_function(
    ctx: &Context,
    function: Ptr<Operation>,
) -> Result<TranslatedFunction, ExportError> {
    let func = function.deref(ctx);
    if func.num_regions() != 1
        || func.get_num_operands() != 0
        || func.get_num_results() != 0
        || func.get_num_successors() != 0
    {
        return Err(reject(
            "function",
            "expected one region and no operands, results, or successors",
        ));
    }
    check_attributes(
        &func.attributes,
        &["builtin_sym_name", "mir_func_type"],
        "function",
    )?;
    get_attribute::<IdentifierAttr>(&func.attributes, "builtin_sym_name", "function")?;
    let type_attr = get_attribute::<TypeAttr>(&func.attributes, "mir_func_type", "function")?;
    let fn_type = type_attr.get_type(ctx).deref(ctx);
    let fn_type = fn_type.downcast_ref::<FunctionType>().ok_or_else(|| {
        reject(
            "function",
            "mir_func_type must contain a builtin.function type",
        )
    })?;
    let inputs = fn_type.arg_types();
    let outputs = fn_type.res_types();
    if outputs.len() != 1 {
        return Err(reject("function signature", "expected exactly one result"));
    }
    let value_type = outputs[0];
    let width = unsigned_width(ctx, value_type)?;
    if inputs.iter().any(|ty| *ty != value_type) {
        return Err(reject(
            "function signature",
            "all arguments must have the same type as the result",
        ));
    }
    let arity = inputs.len();
    let region = func.get_region(0).deref(ctx);
    let blocks: Vec<_> = region.iter(ctx).collect();
    if blocks.len() != 1 {
        return Err(reject(
            "function",
            "expected exactly one nonempty basic block",
        ));
    }
    let block = blocks[0].deref(ctx);
    check_attributes(&block.attributes, &[], "entry block")?;
    if block.get_num_arguments() != arity {
        return Err(reject(
            "entry block",
            "argument count does not match function signature",
        ));
    }

    let mut env = SsaEnv::new();
    let mut let_lines = Vec::new();
    for (index, arg) in block.arguments().enumerate() {
        if arg.get_type(ctx) != value_type {
            return Err(reject(
                "entry block",
                "argument type does not match function signature",
            ));
        }
        let name = env.bind_arg(arg, index, SsaKind::ScalarWord { width });
        let_lines.push(format!(
            "  let {name} : CudaOxide.Mir.Scalar.Expr {width} {arity} := .arg ⟨{index}, by decide⟩"
        ));
    }

    let mut returned_var = None;
    for (index, op_ptr) in block.iter(ctx).enumerate() {
        let opid = Operation::get_opid(op_ptr, ctx).to_string();
        let scope = format!("operation {index} ({opid})");
        if returned_var.is_some() {
            return Err(reject(&scope, "operation after mir.return"));
        }
        let (expected_operands, expected_results, constant_key) = match opid.as_str() {
            "mir.add" | "mir.sub" | "mir.mul" => (2, 1, None),
            "mir.constant" => (0, 1, Some("value")),
            "builtin.constant" => (0, 1, Some("builtin_constant_value")),
            "mir.return" => (1, 0, None),
            _ => return Err(reject(&scope, "unsupported operation in scalar subset")),
        };
        let op = op_ptr.deref(ctx);
        if op.num_regions() != 0 || op.get_num_successors() != 0 {
            return Err(reject(&scope, "regions and successors are unsupported"));
        }
        if op.get_num_operands() != expected_operands || op.get_num_results() != expected_results {
            return Err(reject(&scope, "incorrect operand or result count"));
        }
        check_attributes(
            &op.attributes,
            &constant_key.into_iter().collect::<Vec<_>>(),
            &scope,
        )?;
        let mut operands = Vec::new();
        for operand in op.operands() {
            let binding = env.get(&operand, &scope)?;
            if binding.kind != (SsaKind::ScalarWord { width }) {
                return Err(reject(&scope, "operand type mismatch in scalar expression"));
            }
            operands.push(binding.lean_name.clone());
        }
        for result in op.results() {
            if result.get_type(ctx) != value_type {
                return Err(reject(
                    &scope,
                    "result type does not match function signature",
                ));
            }
        }
        if opid == "mir.return" {
            returned_var = Some(operands[0].clone());
            continue;
        }
        let expression = if let Some(key) = constant_key {
            let literal = get_attribute::<IntegerAttr>(&op.attributes, key, &scope)?;
            if TypeHandle::from(literal.get_type()) != value_type
                || literal.value().bw() != width as usize
            {
                return Err(reject(
                    &scope,
                    "constant type/bit width does not match its result",
                ));
            }
            format!(
                ".literal (BitVec.ofNat {width} {})",
                literal.value().to_string_unsigned_decimal()
            )
        } else {
            let constructor = match opid.as_str() {
                "mir.add" => "add",
                "mir.sub" => "sub",
                "mir.mul" => "mul",
                _ => return Err(reject(&scope, "unsupported arithmetic operation")),
            };
            format!(".{constructor} {} {}", operands[0], operands[1])
        };
        let name = env.bind_result(op.get_result(0), SsaKind::ScalarWord { width });
        let_lines.push(format!(
            "  let {name} : CudaOxide.Mir.Scalar.Expr {width} {arity} := {expression}"
        ));
    }
    let returned_var =
        returned_var.ok_or_else(|| reject("function", "missing mir.return terminator"))?;

    verify_operation(function, ctx)
        .map_err(|error| ExportError::Verification(error.to_string()))?;

    Ok(TranslatedFunction::Scalar {
        width,
        arity,
        let_lines,
        returned_var,
    })
}

/// Translate a `#[verify(...)]` GPU kernel `mir.func` at `PostMem2Reg` into a
/// 1D [`TranslatedFunction::Kernel`] or phased [`TranslatedFunction::TiledKernel2D`].
fn translate_verified_kernel(
    ctx: &Context,
    function: Ptr<Operation>,
) -> Result<TranslatedFunction, ExportError> {
    verify_operation(function, ctx)
        .map_err(|error| ExportError::Verification(error.to_string()))?;

    let (arity, spec) = parse_kernel_verification_spec(ctx, function)?;
    if spec.domain == 2
        || !spec.no_oob_shared.is_empty()
        || !spec.no_data_race.is_empty()
        || spec.uniform_barriers
    {
        return translate_verified_2d_tiled_kernel(ctx, function, spec);
    }
    let func = function.deref(ctx);
    let region = func.get_region(0).deref(ctx);
    let blocks: Vec<Ptr<BasicBlock>> = region.iter(ctx).collect();
    let Some(&entry_block) = blocks.first() else {
        return Err(reject("kernel", "empty kernel region"));
    };

    let mut env = SsaEnv::new();
    for (idx, arg) in entry_block.deref(ctx).arguments().enumerate() {
        env.bind_arg(arg, idx, SsaKind::BufferParam(idx));
    }

    let mut reachable: BTreeSet<usize> = BTreeSet::new();
    reachable.insert(0);
    let mut edge_guards: HashMap<(usize, usize), Option<usize>> = HashMap::new();
    let mut block_arg_incomings: HashMap<(usize, usize), SsaKind> = HashMap::new();

    for _pass in 0..blocks.len() + 2 {
        let prev_env_len = env.len();
        let prev_reachable_len = reachable.len();
        let prev_edges = edge_guards.clone();
        let prev_incomings = block_arg_incomings.clone();

        for (block_idx, &block_ptr) in blocks.iter().enumerate() {
            if !reachable.contains(&block_idx) {
                continue;
            }
            let current_guard = compute_block_guard(block_idx, &edge_guards);
            let block = block_ptr.deref(ctx);

            if block_idx != 0 && block.get_num_arguments() == 1 {
                let some_info = block_arg_incomings
                    .iter()
                    .filter(|((_, dst), _)| *dst == block_idx)
                    .find_map(|(_, incoming)| match *incoming {
                        SsaKind::OptionSome {
                            param,
                            offset,
                            guard_param,
                        } => Some((param, offset, guard_param)),
                        _ => None,
                    });
                if let Some((param, offset, guard_param)) = some_info {
                    env.bind_result(
                        block.get_argument(0),
                        SsaKind::OptionJoin {
                            param,
                            offset,
                            guard_param,
                        },
                    );
                }
            }

            let _ = translate_kernel_block_ops(
                ctx,
                block_idx,
                block_ptr,
                &blocks,
                arity,
                &spec,
                current_guard,
                &mut env,
                &mut reachable,
                &mut edge_guards,
                &mut block_arg_incomings,
                None,
                None,
            );
        }

        if env.len() == prev_env_len
            && reachable.len() == prev_reachable_len
            && edge_guards == prev_edges
            && block_arg_incomings == prev_incomings
        {
            break;
        }
    }

    let mut accesses = Vec::new();
    let mut proven_assert_ops = Vec::new();
    for (block_idx, &block_ptr) in blocks.iter().enumerate() {
        if !reachable.contains(&block_idx) {
            continue;
        }
        let block = block_ptr.deref(ctx);
        if block_idx != 0 && block.get_num_arguments() > 1 {
            return Err(reject(
                "kernel CFG",
                "only single-argument Option join blocks are supported in v0",
            ));
        }
        let current_guard = compute_block_guard(block_idx, &edge_guards);
        translate_kernel_block_ops(
            ctx,
            block_idx,
            block_ptr,
            &blocks,
            arity,
            &spec,
            current_guard,
            &mut env,
            &mut reachable,
            &mut edge_guards,
            &mut block_arg_incomings,
            Some(&mut accesses),
            Some(&mut proven_assert_ops),
        )?;
    }

    Ok(TranslatedFunction::Kernel {
        arity,
        spec,
        accesses,
        proven_assert_ops,
    })
}

/// Extract the numeric index from a `mir.field_index` or enum-variant attribute
/// named `key_name` on `attrs`.
fn attribute_field_index(attrs: &AttributeDict, key_name: &str) -> Option<u32> {
    let text = attrs
        .0
        .iter()
        .find(|(k, _)| k.to_string() == key_name)
        .map(|(_, v)| format!("{v:?}"))?;
    text.chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse::<u32>()
        .ok()
}

/// Compute the dominating `< param.len()` guard parameter index for `block_idx`
/// when every incoming CFG edge into `block_idx` carries the exact same guard.
fn compute_block_guard(
    block_idx: usize,
    edge_guards: &HashMap<(usize, usize), Option<usize>>,
) -> Option<usize> {
    if block_idx == 0 {
        return None;
    }
    let mut incoming = edge_guards
        .iter()
        .filter(|((_, dst), _)| *dst == block_idx)
        .map(|(_, guard)| *guard);
    let first = incoming.next()??;
    if incoming.all(|g| g == Some(first)) {
        Some(first)
    } else {
        None
    }
}

/// Walk the operations of a single basic block in a 1D verified kernel, updating
/// [`SsaEnv`], outgoing CFG edge guards, and (on the final pass) recording
/// guarded buffer accesses and proven `mir.assert` nodes.
#[allow(clippy::too_many_arguments)]
fn translate_kernel_block_ops(
    ctx: &Context,
    block_idx: usize,
    block_ptr: Ptr<BasicBlock>,
    blocks: &[Ptr<BasicBlock>],
    arity: usize,
    spec: &KernelVerificationSpec,
    current_guard: Option<usize>,
    env: &mut SsaEnv,
    reachable: &mut BTreeSet<usize>,
    edge_guards: &mut HashMap<(usize, usize), Option<usize>>,
    block_arg_incomings: &mut HashMap<(usize, usize), SsaKind>,
    mut record_accesses: Option<&mut Vec<ExtractedAccess>>,
    mut record_proven_asserts: Option<&mut Vec<Ptr<Operation>>>,
) -> Result<(), ExportError> {
    let block = block_ptr.deref(ctx);
    for op_ptr in block.iter(ctx) {
        let opid = Operation::get_opid(op_ptr, ctx).to_string();
        let op = op_ptr.deref(ctx);
        match opid.as_str() {
            "mir.storage_live" | "mir.storage_dead" | "mir.dbg_value" | "mir.dbg_declare" => {}
            "mir.float_constant" => {
                if op.get_num_results() == 1 {
                    env.bind_result(op.get_result(0), SsaKind::DataValue);
                }
            }
            _ if opid.starts_with("nvvm.shfl_sync_") => {
                if op.get_num_results() == 1 {
                    env.bind_result(op.get_result(0), SsaKind::DataValue);
                }
            }
            "mir.alloca" => {
                if op.get_num_results() == 1 {
                    env.bind_result(op.get_result(0), SsaKind::LaunchContextAlloca);
                }
            }
            "mir.cast" => {
                if op.get_num_operands() == 1
                    && op.get_num_results() == 1
                    && let Some(kind) = env.get_kind(&op.get_operand(0))
                {
                    env.bind_result(op.get_result(0), kind);
                }
            }
            "mir.call" => {
                let callee = get_string_attr(&op.attributes, "callee").unwrap_or_default();
                if callee.contains("index_1d") && op.get_num_results() == 1 {
                    env.bind_result(op.get_result(0), SsaKind::ThreadIndexStruct);
                } else if (callee.contains("warp__reduce_") || callee.contains("warp__shuffle_"))
                    && op.get_num_results() == 1
                {
                    env.bind_result(op.get_result(0), SsaKind::DataValue);
                } else {
                    return Err(reject(
                        "mir.call",
                        format!("unsupported call `{callee}` in verified kernel"),
                    ));
                }
            }
            "mir.extract_field" => {
                let base = env.get_kind(&op.get_operand(0));
                let field = attribute_field_index(&op.attributes, "index");
                let mapped = match (base, field) {
                    (Some(SsaKind::ThreadIndexStruct), Some(0)) => {
                        Some(SsaKind::Index1D { offset: 0 })
                    }
                    (Some(SsaKind::CheckedIndex1D { offset }), Some(0)) => {
                        Some(SsaKind::Index1D { offset })
                    }
                    (Some(SsaKind::CheckedIndex1D { .. }), Some(1)) => Some(SsaKind::DataValue),
                    (Some(SsaKind::BufferParam(p)), Some(0)) if p < arity => {
                        Some(SsaKind::SlicePtr(p))
                    }
                    (Some(SsaKind::BufferParam(p)), Some(1)) if p < arity => {
                        Some(SsaKind::SliceLen(p))
                    }
                    _ => None,
                };
                if let Some(kind) = mapped {
                    env.bind_result(op.get_result(0), kind);
                } else {
                    return Err(reject(
                        "mir.extract_field",
                        "unsupported field extraction in verified kernel",
                    ));
                }
            }
            "mir.constant" | "builtin.constant" => {
                let key = if opid == "mir.constant" {
                    "value"
                } else {
                    "builtin_constant_value"
                };
                if let Some(int_attr) = op
                    .attributes
                    .0
                    .iter()
                    .find(|(k, _)| k.to_string() == key)
                    .and_then(|(_, v)| v.downcast_ref::<IntegerAttr>())
                {
                    let value = int_attr
                        .value()
                        .to_string_unsigned_decimal()
                        .parse::<u128>()
                        .unwrap_or(0);
                    let width = u32::try_from(int_attr.value().bw()).unwrap_or(0);
                    env.bind_result(op.get_result(0), SsaKind::ConstInt { width, value });
                } else {
                    env.bind_result(op.get_result(0), SsaKind::DataValue);
                }
            }
            "mir.checked_add" => {
                let lhs = env.get_kind(&op.get_operand(0));
                let rhs = env.get_kind(&op.get_operand(1));
                match (lhs, rhs) {
                    (
                        Some(SsaKind::Index1D { offset }),
                        Some(SsaKind::ConstInt { value: c, .. }),
                    )
                    | (
                        Some(SsaKind::ConstInt { value: c, .. }),
                        Some(SsaKind::Index1D { offset }),
                    ) => {
                        let new_offset =
                            offset.saturating_add(u64::try_from(c).unwrap_or(u64::MAX));
                        env.bind_result(
                            op.get_result(0),
                            SsaKind::CheckedIndex1D { offset: new_offset },
                        );
                    }
                    _ => {
                        return Err(reject(
                            "mir.checked_add",
                            "unsupported checked_add operands in index expression",
                        ));
                    }
                }
            }
            "mir.add" | "mir.sub" | "mir.mul" => {
                let lhs = env.get_kind(&op.get_operand(0));
                let rhs = env.get_kind(&op.get_operand(1));
                if opid == "mir.add" {
                    match (&lhs, &rhs) {
                        (
                            Some(SsaKind::Index1D { offset }),
                            Some(SsaKind::ConstInt { value: c, .. }),
                        )
                        | (
                            Some(SsaKind::ConstInt { value: c, .. }),
                            Some(SsaKind::Index1D { offset }),
                        ) => {
                            let new_offset =
                                offset.saturating_add(u64::try_from(*c).unwrap_or(u64::MAX));
                            env.bind_result(
                                op.get_result(0),
                                SsaKind::Index1D { offset: new_offset },
                            );
                            continue;
                        }
                        _ => {}
                    }
                }
                env.bind_result(op.get_result(0), SsaKind::DataValue);
            }
            "mir.eq" => {
                let lhs = env.get_kind(&op.get_operand(0));
                let rhs = env.get_kind(&op.get_operand(1));
                let mapped = match (lhs, rhs) {
                    (
                        Some(SsaKind::Index1D { offset: 0 }),
                        Some(SsaKind::ConstInt { value: u64_max, .. }),
                    ) if u64_max == u128::from(u64::MAX) => SsaKind::IsSentinel,
                    (
                        Some(SsaKind::OptionDiscriminant {
                            param,
                            offset,
                            guard_param,
                        }),
                        Some(SsaKind::ConstInt { value: 1, .. }),
                    ) => SsaKind::OptionIsSome {
                        param,
                        offset,
                        guard_param,
                    },
                    (
                        Some(SsaKind::OptionDiscriminant { .. }),
                        Some(SsaKind::ConstInt { value: 0, .. }),
                    ) => SsaKind::OptionIsNone,
                    _ => SsaKind::DataValue,
                };
                env.bind_result(op.get_result(0), mapped);
            }
            "mir.lt" => {
                let lhs = env.get_kind(&op.get_operand(0));
                let rhs = env.get_kind(&op.get_operand(1));
                let mapped = match (lhs, rhs) {
                    (
                        Some(SsaKind::Index1D { offset: 0 }),
                        Some(SsaKind::SliceLen(guard_param)),
                    ) => SsaKind::IndexLtLen { guard_param },
                    _ => SsaKind::DataValue,
                };
                env.bind_result(op.get_result(0), mapped);
            }
            "mir.not" => {
                let inner = env.get_kind(&op.get_operand(0));
                let mapped = match inner {
                    Some(SsaKind::IndexLtLen { guard_param }) => {
                        SsaKind::IndexGeLen { guard_param }
                    }
                    Some(SsaKind::IndexGeLen { guard_param }) => {
                        SsaKind::IndexLtLen { guard_param }
                    }
                    _ => SsaKind::DataValue,
                };
                env.bind_result(op.get_result(0), mapped);
            }
            "mir.ptr_offset" => {
                let base = env.get_kind(&op.get_operand(0));
                let idx = env.get_kind(&op.get_operand(1));
                match (base, idx) {
                    (Some(SsaKind::SlicePtr(param)), Some(SsaKind::Index1D { offset })) => {
                        env.bind_result(op.get_result(0), SsaKind::ElementPtr { param, offset });
                    }
                    _ => {
                        return Err(reject(
                            "mir.ptr_offset",
                            "pointer offset must index a kernel slice parameter by the 1D thread index",
                        ));
                    }
                }
            }
            "mir.construct_enum" => {
                let variant = attribute_field_index(&op.attributes, "construct_enum_variant_index");
                match variant {
                    Some(0) => {
                        env.bind_result(op.get_result(0), SsaKind::OptionNone);
                    }
                    Some(1) if op.get_num_operands() == 1 => {
                        let Some(SsaKind::ElementPtr { param, offset }) =
                            env.get_kind(&op.get_operand(0))
                        else {
                            return Err(reject(
                                "mir.construct_enum",
                                "Option::Some payload must be a slice element pointer",
                            ));
                        };
                        let Some(guard_param) = current_guard else {
                            return Err(reject(
                                "mir.construct_enum",
                                "Option::Some slice element pointer constructed without a dominating bounds guard",
                            ));
                        };
                        env.bind_result(
                            op.get_result(0),
                            SsaKind::OptionSome {
                                param,
                                offset,
                                guard_param,
                            },
                        );
                    }
                    _ => {
                        return Err(reject(
                            "mir.construct_enum",
                            "unsupported enum construction in verified kernel",
                        ));
                    }
                }
            }
            "mir.get_discriminant" => {
                if let Some(SsaKind::OptionJoin {
                    param,
                    offset,
                    guard_param,
                }) = env.get_kind(&op.get_operand(0))
                {
                    env.bind_result(
                        op.get_result(0),
                        SsaKind::OptionDiscriminant {
                            param,
                            offset,
                            guard_param,
                        },
                    );
                } else {
                    return Err(reject(
                        "mir.get_discriminant",
                        "unsupported discriminant operand in verified kernel",
                    ));
                }
            }
            "mir.enum_payload" => {
                if let Some(SsaKind::OptionJoin { param, offset, .. }) =
                    env.get_kind(&op.get_operand(0))
                {
                    env.bind_result(op.get_result(0), SsaKind::ElementPtr { param, offset });
                } else {
                    return Err(reject(
                        "mir.enum_payload",
                        "unsupported enum payload extraction in verified kernel",
                    ));
                }
            }
            "mir.assert" => {
                if let Some(proven_asserts) = record_proven_asserts.as_deref_mut()
                    && current_guard.is_some()
                    && op.get_num_operands() == 1
                    && let Some(SsaKind::IndexLtLen {
                        guard_param: checked_param,
                    }) = env.get_kind(&op.get_operand(0))
                    && spec.no_oob_idx.contains(&checked_param)
                {
                    proven_asserts.push(op_ptr);
                }
            }
            "mir.load" => {
                let Some(SsaKind::ElementPtr { param, offset }) = env.get_kind(&op.get_operand(0))
                else {
                    return Err(reject(
                        "mir.load",
                        "memory load does not target a recognized slice element pointer",
                    ));
                };
                let Some(guard_param) = current_guard else {
                    return Err(reject(
                        "mir.load",
                        format!(
                            "unguarded read from parameter {param}: every buffer read must be dominated by a strict `< len` guard"
                        ),
                    ));
                };
                if let Some(accesses) = record_accesses.as_deref_mut() {
                    accesses.push(ExtractedAccess {
                        kind: AccessKind::Read,
                        target_param: param,
                        guard_param,
                        index_offset: offset,
                    });
                }
                env.bind_result(op.get_result(0), SsaKind::DataValue);
            }
            "mir.store" => {
                let Some(SsaKind::ElementPtr { param, offset }) = env.get_kind(&op.get_operand(0))
                else {
                    return Err(reject(
                        "mir.store",
                        "memory store does not target a recognized slice element pointer",
                    ));
                };
                let Some(guard_param) = current_guard else {
                    return Err(reject(
                        "mir.store",
                        format!(
                            "unguarded write to parameter {param}: every buffer write must be dominated by a strict `< len` guard"
                        ),
                    ));
                };
                if let Some(accesses) = record_accesses.as_deref_mut() {
                    accesses.push(ExtractedAccess {
                        kind: AccessKind::Write,
                        target_param: param,
                        guard_param,
                        index_offset: offset,
                    });
                }
            }
            "mir.goto" => {
                let target = op.get_successor(0);
                if let Some(dst_idx) = blocks.iter().position(|&b| b == target) {
                    reachable.insert(dst_idx);
                    edge_guards.insert((block_idx, dst_idx), current_guard);
                    if op.get_num_operands() == 1
                        && let Some(kind) = env.get_kind(&op.get_operand(0))
                    {
                        block_arg_incomings.insert((block_idx, dst_idx), kind);
                    }
                }
            }
            "mir.cond_br" => {
                let cond = env.get_kind(&op.get_operand(0));
                let true_succ = op.get_successor(0);
                let false_succ = op.get_successor(1);
                let (true_guard, false_guard) = match cond {
                    Some(SsaKind::IndexLtLen { guard_param }) => (Some(guard_param), current_guard),
                    Some(SsaKind::IndexGeLen { guard_param }) => (current_guard, Some(guard_param)),
                    Some(SsaKind::OptionIsSome { guard_param, .. }) => {
                        (Some(guard_param), current_guard)
                    }
                    _ => (current_guard, current_guard),
                };
                if let Some(dst_idx) = blocks.iter().position(|&b| b == true_succ) {
                    reachable.insert(dst_idx);
                    edge_guards.insert((block_idx, dst_idx), true_guard);
                }
                if let Some(dst_idx) = blocks.iter().position(|&b| b == false_succ) {
                    reachable.insert(dst_idx);
                    edge_guards.insert((block_idx, dst_idx), false_guard);
                }
            }
            "mir.return" | "mir.unreachable" => {}
            _ => {
                return Err(reject(
                    &opid,
                    format!("unsupported operation `{opid}` in verified kernel subset"),
                ));
            }
        }
    }
    Ok(())
}

/// Extract a signed/unsigned integer attribute by key as `u64`.
fn int_attr_u64(attrs: &AttributeDict, key: &str) -> Option<u64> {
    attrs
        .0
        .iter()
        .find(|(k, _)| k.to_string() == key)
        .and_then(|(_, v)| v.downcast_ref::<IntegerAttr>())
        .and_then(|attr| {
            attr.value()
                .to_string_unsigned_decimal()
                .parse::<u64>()
                .ok()
        })
}

/// Extract a compositional [`IdxExpr`] from an [`SsaKind`] when the operand is
/// an index expression or an integer constant literal.
fn as_idx_expr(kind: &SsaKind) -> Option<IdxExpr> {
    match kind {
        SsaKind::Index(expr) => Some(expr.clone()),
        SsaKind::ConstInt { value, .. } => u64::try_from(*value).ok().map(IdxExpr::Const),
        _ => None,
    }
}

/// Return true if `op` is a `mir.shared_alloc` for a 64-bit hardware `Barrier` (`mbarrier`).
fn is_mbarrier_shared_alloc(ctx: &Context, op: &Operation) -> bool {
    let size = int_attr_u64(&op.attributes, "size").unwrap_or(0);
    if size != 1 {
        return false;
    }
    op.attributes.0.iter().any(|(k, v)| {
        k.to_string() == "elem_type"
            && v.downcast_ref::<TypeAttr>()
                .and_then(|ta| {
                    ta.get_type(ctx)
                        .deref(ctx)
                        .downcast_ref::<IntegerType>()
                        .map(|i| i.width() == 64)
                })
                .unwrap_or(false)
    })
}

/// Check if `start` can reach `target` in `succs` without passing through `forbidden`.
fn can_reach_without(
    start: usize,
    target: usize,
    forbidden: usize,
    succs: &[Vec<usize>],
    num_blocks: usize,
) -> bool {
    if start == forbidden {
        return false;
    }
    let mut visited = vec![false; num_blocks];
    let mut stack = vec![start];
    visited[start] = true;
    while let Some(curr) = stack.pop() {
        if curr == target {
            return true;
        }
        for &next in &succs[curr] {
            if next != forbidden && !visited[next] {
                visited[next] = true;
                stack.push(next);
            }
        }
    }
    false
}

/// Check if `start` can reach `target` in `succs`.
fn can_reach(start: usize, target: usize, succs: &[Vec<usize>], num_blocks: usize) -> bool {
    can_reach_without(start, target, usize::MAX, succs, num_blocks)
}

/// Translate a 2D phased shared-memory kernel (`sgemm_tiled_views`) using `SsaEnv`.
fn translate_verified_2d_tiled_kernel(
    ctx: &Context,
    function: Ptr<Operation>,
    spec: KernelVerificationSpec,
) -> Result<TranslatedFunction, ExportError> {
    let func = function.deref(ctx);
    let region = func.get_region(0).deref(ctx);
    let blocks: Vec<Ptr<BasicBlock>> = region.iter(ctx).collect();
    if blocks.is_empty() {
        return Err(reject("kernel", "empty kernel region"));
    }

    // 1. Discover all static `SharedArray` and `Barrier` (`mbarrier`) allocations
    //    (`mir.shared_alloc`) and assign deterministic `0..N` IDs sorted by short name.
    let mut shared_raw_list: Vec<(String, u64)> = Vec::new();
    let mut bar_raw_list: Vec<String> = Vec::new();
    for &b_ptr in &blocks {
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            if Operation::get_opid(op_ptr, ctx).to_string() == "mir.shared_alloc" {
                let op = op_ptr.deref(ctx);
                let full_name = get_string_attr(&op.attributes, "source_name")
                    .unwrap_or_else(|| "SHARED".to_string());
                let short = full_name
                    .rsplit("::")
                    .next()
                    .unwrap_or(&full_name)
                    .to_string();
                if is_mbarrier_shared_alloc(ctx, &op) {
                    if !bar_raw_list.contains(&short) {
                        bar_raw_list.push(short);
                    }
                } else {
                    let size = int_attr_u64(&op.attributes, "size").unwrap_or(0);
                    if !shared_raw_list.iter().any(|(n, _)| n == &short) {
                        shared_raw_list.push((short, size));
                    }
                }
            }
        }
    }
    shared_raw_list.sort_by(|a, b| a.0.cmp(&b.0));
    bar_raw_list.sort();

    // 2. Build the block adjacency list, locate `nvvm.barrier0` blocks, and
    //    find `mir.return` blocks.
    let num_blocks = blocks.len();
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); num_blocks];
    let mut barrier_blocks: Vec<usize> = Vec::new();
    let mut return_blocks: Vec<usize> = Vec::new();

    for (b_idx, &b_ptr) in blocks.iter().enumerate() {
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            let opid = Operation::get_opid(op_ptr, ctx).to_string();
            let op = op_ptr.deref(ctx);
            if opid == "nvvm.barrier0" {
                barrier_blocks.push(b_idx);
            } else if opid == "mir.return" {
                return_blocks.push(b_idx);
            } else if opid == "mir.goto" || opid == "mir.cond_br" {
                for s_idx in 0..op.get_num_successors() {
                    let target = op.get_successor(s_idx);
                    if let Some(dst) = blocks.iter().position(|&b| b == target)
                        && !succs[b_idx].contains(&dst)
                    {
                        succs[b_idx].push(dst);
                    }
                }
            }
        }
    }

    // 3. Check barrier uniformity and assign a barrier phase to each block.
    //    - In a 2-barrier tile loop (`Phase 0 -> barrier0 -> Phase 1 -> barrier1 -> backedge`),
    //      blocks before `barrier0` are in Phase 0, and blocks after `barrier0` that cannot
    //      be reached without crossing `barrier0` (and sit before `barrier1`) are in Phase 1.
    //    - If a barrier is missing (`barrier_blocks.len() < 2`), the phases collapse into Phase 0
    //      because either there is no barrier between write and read, or the read wraps around
    //      the loop backedge into the next iteration's write without a barrier!
    let mut barriers_uniform = !barrier_blocks.is_empty();
    let mut enclosing_loop_header = None;
    if let Some(&first_barrier) = barrier_blocks.first() {
        // Find the loop header that controls the barrier loop (a block that reaches
        // `first_barrier` and is also reachable from `first_barrier`).
        let loop_header = (0..num_blocks).find(|&h| {
            h != first_barrier
                && can_reach(h, first_barrier, &succs, num_blocks)
                && can_reach(first_barrier, h, &succs, num_blocks)
                && !can_reach_without(0, first_barrier, h, &succs, num_blocks)
        });
        enclosing_loop_header = loop_header;
        if let Some(header) = loop_header {
            // Condition A: Entry block 0 must not be able to reach any `mir.return`
            // without passing through `header` (catches early `if row >= m { return; }`).
            for &ret_blk in &return_blocks {
                if can_reach_without(0, ret_blk, header, &succs, num_blocks) {
                    barriers_uniform = false;
                }
            }
            // Condition B: Every barrier in the loop must lie on ALL paths from the
            // loop body entry back to `header` (catches `if tx == 0 { sync_threads() }`).
            for &b_blk in &barrier_blocks {
                for &body_start in &succs[header] {
                    if can_reach(body_start, header, &succs, num_blocks)
                        && can_reach_without(body_start, header, b_blk, &succs, num_blocks)
                    {
                        barriers_uniform = false;
                    }
                }
            }
        } else {
            // No enclosing loop: every barrier must lie on all paths from entry `0` to `return`.
            for &b_blk in &barrier_blocks {
                for &ret_blk in &return_blocks {
                    if can_reach_without(0, ret_blk, b_blk, &succs, num_blocks) {
                        barriers_uniform = false;
                    }
                }
            }
        }
    }

    let block_phase = |b_idx: usize, seen_barriers_in_block: usize| -> usize {
        if !barriers_uniform || barrier_blocks.is_empty() {
            return 0;
        }
        let b0 = barrier_blocks[0];
        if enclosing_loop_header.is_some() {
            if barrier_blocks.len() >= 2 {
                let b1 = barrier_blocks[1];
                if b_idx == b0 {
                    return usize::from(seen_barriers_in_block > 0);
                }
                if can_reach_without(b0, b_idx, b1, &succs, num_blocks) {
                    return 1;
                }
            }
            0
        } else {
            // Acyclic N-barrier kernel (`Phase 0 -> sync_threads() -> Phase 1 -> ... -> Phase N`):
            // Count every barrier in a distinct block `bk` that strictly dominates `b_idx`
            // from entry `0`, plus any `nvvm.barrier0` ops already passed inside `b_idx`.
            let dominating_barriers = barrier_blocks
                .iter()
                .filter(|&&bk| {
                    bk != b_idx
                        && can_reach(bk, b_idx, &succs, num_blocks)
                        && !can_reach_without(0, b_idx, bk, &succs, num_blocks)
                })
                .count();
            dominating_barriers + seen_barriers_in_block
        }
    };

    // 4. Fixed-point SSA propagation over the function blocks.
    //    Blocks are not in topological order (e.g., `^block73v1` defines `tid.x`
    //    after `^block59v1` uses it), so we iterate until all `SsaKind` bindings
    //    stabilize before enforcing fail-closed checks on `mir.ptr_offset`.
    let mut env = SsaEnv::new();
    if let Some(&entry_block) = blocks.first() {
        for (idx, arg) in entry_block.deref(ctx).arguments().enumerate() {
            env.bind_arg(arg, idx, SsaKind::BufferParam(idx));
        }
    }
    for _pass in 0..num_blocks + 3 {
        let mut changed = false;
        let mut update = |env: &mut SsaEnv, val: pliron::value::Value, kind: SsaKind| {
            let old = env.get_kind(&val);
            if old.as_ref() != Some(&kind) {
                // Never downgrade a specific kind back to `DataValue`.
                if kind == SsaKind::DataValue && old.is_some() {
                    return;
                }
                env.bind_result(val, kind);
                changed = true;
            }
        };
        for (b_idx, &b_ptr) in blocks.iter().enumerate() {
            let block = b_ptr.deref(ctx);
            if b_idx != 0 {
                for arg in block.arguments() {
                    update(&mut env, arg, SsaKind::DataValue);
                }
            }
            // Detect inner scan loop counter `i < bound` on block arguments.
            for step_arg in block.arguments() {
                for op_ptr in block.iter(ctx) {
                    if Operation::get_opid(op_ptr, ctx).to_string() == "mir.lt" {
                        let op = op_ptr.deref(ctx);
                        if op.get_num_operands() == 2
                            && op.get_operand(0) == step_arg
                            && let Some(SsaKind::ConstInt { value: bound, .. }) =
                                env.get_kind(&op.get_operand(1))
                        {
                            update(
                                &mut env,
                                step_arg,
                                SsaKind::Index(IdxExpr::LoopVar {
                                    bound: u64::try_from(bound).unwrap_or(u64::MAX),
                                }),
                            );
                        }
                    }
                }
            }
            for op_ptr in block.iter(ctx) {
                let opid = Operation::get_opid(op_ptr, ctx).to_string();
                let op = op_ptr.deref(ctx);
                match opid.as_str() {
                    "nvvm.read_ptx_sreg_tid_x" => {
                        update(&mut env, op.get_result(0), SsaKind::Index(IdxExpr::Tx));
                    }
                    "nvvm.read_ptx_sreg_tid_y" => {
                        update(&mut env, op.get_result(0), SsaKind::Index(IdxExpr::Ty));
                    }
                    "nvvm.read_ptx_sreg_laneid" => {
                        update(
                            &mut env,
                            op.get_result(0),
                            SsaKind::Index(IdxExpr::Mod(Box::new(IdxExpr::Tx), 32)),
                        );
                    }
                    "mir.call" => {
                        let callee = get_string_attr(&op.attributes, "callee").unwrap_or_default();
                        if callee.contains("lane_id") && op.get_num_results() == 1 {
                            update(
                                &mut env,
                                op.get_result(0),
                                SsaKind::Index(IdxExpr::Mod(Box::new(IdxExpr::Tx), 32)),
                            );
                        } else if op.get_num_results() == 1 {
                            update(&mut env, op.get_result(0), SsaKind::DataValue);
                        }
                    }
                    "mir.div" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(SsaKind::ConstInt { value: g, .. })) =
                            (lhs.as_ref().and_then(as_idx_expr), rhs)
                            && g > 0
                            && a.has_coord()
                        {
                            SsaKind::Index(IdxExpr::Div(
                                Box::new(a),
                                u64::try_from(g).unwrap_or(u64::MAX),
                            ))
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.shr" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(SsaKind::ConstInt { value: sh, .. })) =
                            (lhs.as_ref().and_then(as_idx_expr), rhs)
                            && sh < 16
                            && a.has_coord()
                        {
                            SsaKind::Index(IdxExpr::Div(Box::new(a), 1u64 << sh))
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.rem" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(SsaKind::ConstInt { value: g, .. })) =
                            (lhs.as_ref().and_then(as_idx_expr), rhs)
                            && g > 0
                            && a.has_coord()
                        {
                            SsaKind::Index(IdxExpr::Mod(
                                Box::new(a),
                                u64::try_from(g).unwrap_or(u64::MAX),
                            ))
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.bitand" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = match (&lhs, &rhs) {
                            (Some(l), Some(SsaKind::ConstInt { value: m, .. }))
                            | (Some(SsaKind::ConstInt { value: m, .. }), Some(l))
                                if *m > 0 && (*m + 1).is_power_of_two() =>
                            {
                                if let Some(a) = as_idx_expr(l)
                                    && a.has_coord()
                                {
                                    SsaKind::Index(IdxExpr::Mod(
                                        Box::new(a),
                                        u64::try_from(*m + 1).unwrap_or(u64::MAX),
                                    ))
                                } else {
                                    SsaKind::DataValue
                                }
                            }
                            _ => SsaKind::DataValue,
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.eq" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(b)) = (
                            lhs.as_ref().and_then(as_idx_expr),
                            rhs.as_ref().and_then(as_idx_expr),
                        ) && (a.has_coord() || b.has_coord())
                        {
                            SsaKind::Guard(GuardExpr::Eq(a, b))
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.ne" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(b)) = (
                            lhs.as_ref().and_then(as_idx_expr),
                            rhs.as_ref().and_then(as_idx_expr),
                        ) && (a.has_coord() || b.has_coord())
                        {
                            SsaKind::Guard(GuardExpr::Ne(a, b))
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.not" => {
                        let inner = env.get_kind(&op.get_operand(0));
                        let mapped = match inner {
                            Some(SsaKind::Guard(g)) => SsaKind::Guard(g.negate()),
                            Some(SsaKind::MbarrierReady { bar_id }) => {
                                SsaKind::MbarrierNotReady { bar_id }
                            }
                            Some(SsaKind::MbarrierNotReady { bar_id }) => {
                                SsaKind::MbarrierReady { bar_id }
                            }
                            _ => SsaKind::DataValue,
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "nvvm.mbarrier_try_wait_parity_shared"
                    | "nvvm.mbarrier_try_wait_shared"
                    | "nvvm.mbarrier_test_wait_shared"
                    | "nvvm.mbarrier_try_wait_parity_cluster"
                    | "nvvm.mbarrier_try_wait_parity_cluster_relaxed" => {
                        let mapped = if op.get_num_operands() >= 1
                            && let Some(SsaKind::MbarrierPtr { bar_id }) =
                                env.get_kind(&op.get_operand(0))
                        {
                            SsaKind::MbarrierReady { bar_id }
                        } else {
                            SsaKind::DataValue
                        };
                        if op.get_num_results() == 1 {
                            update(&mut env, op.get_result(0), mapped);
                        }
                    }
                    "mir.ge" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(b)) = (
                            lhs.as_ref().and_then(as_idx_expr),
                            rhs.as_ref().and_then(as_idx_expr),
                        ) && (a.has_coord() || b.has_coord())
                        {
                            SsaKind::Guard(GuardExpr::Ge(a, b))
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.constant" | "builtin.constant" => {
                        let key = if opid == "mir.constant" {
                            "value"
                        } else {
                            "builtin_constant_value"
                        };
                        if let Some(val) = int_attr_u64(&op.attributes, key) {
                            update(
                                &mut env,
                                op.get_result(0),
                                SsaKind::ConstInt {
                                    width: 64,
                                    value: u128::from(val),
                                },
                            );
                        } else if op.get_num_results() == 1 {
                            update(&mut env, op.get_result(0), SsaKind::DataValue);
                        }
                    }
                    "mir.shared_alloc" => {
                        let full_name = get_string_attr(&op.attributes, "source_name")
                            .unwrap_or_else(|| "SHARED".to_string());
                        let short = full_name
                            .rsplit("::")
                            .next()
                            .unwrap_or(&full_name)
                            .to_string();
                        if is_mbarrier_shared_alloc(ctx, &op) {
                            let bar_id = bar_raw_list.iter().position(|n| n == &short).unwrap_or(0);
                            update(&mut env, op.get_result(0), SsaKind::MbarrierPtr { bar_id });
                        } else {
                            let size = int_attr_u64(&op.attributes, "size").unwrap_or(0);
                            let shared_id = shared_raw_list
                                .iter()
                                .position(|(n, _)| n == &short)
                                .unwrap_or(0);
                            update(
                                &mut env,
                                op.get_result(0),
                                SsaKind::SharedBasePtr { shared_id, size },
                            );
                        }
                    }
                    "mir.cast" => {
                        if op.get_num_operands() == 1
                            && op.get_num_results() == 1
                            && let Some(kind) = env.get_kind(&op.get_operand(0))
                        {
                            update(&mut env, op.get_result(0), kind);
                        }
                    }
                    "mir.shl" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(SsaKind::ConstInt { value: sh, .. })) =
                            (lhs.as_ref().and_then(as_idx_expr), rhs)
                            && sh < 16
                            && a.has_coord()
                        {
                            SsaKind::Index(IdxExpr::Mul(
                                Box::new(a),
                                Box::new(IdxExpr::Const(1u64 << sh)),
                            ))
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.mul" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(b)) = (
                            lhs.as_ref().and_then(as_idx_expr),
                            rhs.as_ref().and_then(as_idx_expr),
                        ) && (a.has_coord() || b.has_coord())
                        {
                            SsaKind::Index(IdxExpr::Mul(Box::new(a), Box::new(b)))
                        } else if matches!(
                            (&lhs, &rhs),
                            (
                                Some(SsaKind::Index(IdxExpr::Ty) | SsaKind::StripRowOffset),
                                Some(SsaKind::DataValue | SsaKind::BufferParam(_)),
                            ) | (
                                Some(SsaKind::DataValue | SsaKind::BufferParam(_)),
                                Some(SsaKind::Index(IdxExpr::Ty) | SsaKind::StripRowOffset),
                            )
                        ) {
                            SsaKind::StripColStrided
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.add" | "mir.bitor" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(b)) = (
                            lhs.as_ref().and_then(as_idx_expr),
                            rhs.as_ref().and_then(as_idx_expr),
                        ) && (a.has_coord() || b.has_coord())
                        {
                            SsaKind::Index(IdxExpr::Add(Box::new(a), Box::new(b)))
                        } else if matches!(
                            (&lhs, &rhs),
                            (
                                Some(SsaKind::Index(IdxExpr::Tx) | SsaKind::StripColOffset),
                                Some(SsaKind::DataValue),
                            ) | (
                                Some(SsaKind::DataValue),
                                Some(SsaKind::Index(IdxExpr::Tx) | SsaKind::StripColOffset),
                            )
                        ) {
                            SsaKind::StripColOffset
                        } else if matches!(
                            (&lhs, &rhs),
                            (
                                Some(SsaKind::Index(IdxExpr::Ty) | SsaKind::StripRowOffset),
                                Some(SsaKind::DataValue),
                            ) | (
                                Some(SsaKind::DataValue),
                                Some(SsaKind::Index(IdxExpr::Ty) | SsaKind::StripRowOffset),
                            )
                        ) {
                            SsaKind::StripRowOffset
                        } else if matches!(
                            (&lhs, &rhs),
                            (Some(SsaKind::StripColStrided), Some(SsaKind::DataValue))
                                | (Some(SsaKind::DataValue), Some(SsaKind::StripColStrided))
                        ) {
                            SsaKind::StripColStrided
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.extract_field" => {
                        let base = env.get_kind(&op.get_operand(0));
                        let field = attribute_field_index(&op.attributes, "index");
                        let mapped = match (base, field) {
                            (Some(SsaKind::BufferParam(p)), Some(0)) => SsaKind::SlicePtr(p),
                            (Some(SsaKind::BufferParam(p)), Some(1)) => SsaKind::SliceLen(p),
                            _ => SsaKind::DataValue,
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.lt" => {
                        let lhs = env.get_kind(&op.get_operand(0));
                        let rhs = env.get_kind(&op.get_operand(1));
                        let mapped = if let (Some(a), Some(b)) = (
                            lhs.as_ref().and_then(as_idx_expr),
                            rhs.as_ref().and_then(as_idx_expr),
                        ) && a.has_coord()
                        {
                            SsaKind::Guard(GuardExpr::Lt(a, b))
                        } else if let (
                            Some(SsaKind::StripColOffset | SsaKind::StripColStrided),
                            Some(SsaKind::SliceLen(guard_param)),
                        ) = (lhs, rhs)
                        {
                            SsaKind::IndexLtLen { guard_param }
                        } else {
                            SsaKind::DataValue
                        };
                        update(&mut env, op.get_result(0), mapped);
                    }
                    "mir.ptr_offset" => {
                        let base = env.get_kind(&op.get_operand(0));
                        let idx = env.get_kind(&op.get_operand(1));
                        if let (
                            Some(SsaKind::SharedBasePtr { shared_id, size }),
                            Some(SsaKind::Index(addr)),
                        ) = (base.clone(), idx.clone())
                            && addr.has_coord()
                        {
                            update(
                                &mut env,
                                op.get_result(0),
                                SsaKind::SharedElementPtr {
                                    shared_id,
                                    size,
                                    addr,
                                },
                            );
                        } else if let (
                            Some(SsaKind::SlicePtr(param)),
                            Some(SsaKind::StripColOffset),
                        ) = (base.clone(), idx.clone())
                        {
                            update(
                                &mut env,
                                op.get_result(0),
                                SsaKind::GlobalStripElementPtr {
                                    param,
                                    kind: GlobalStripKind::RowStrip,
                                },
                            );
                        } else if let (
                            Some(SsaKind::SlicePtr(param)),
                            Some(SsaKind::StripColStrided),
                        ) = (base, idx)
                        {
                            update(
                                &mut env,
                                op.get_result(0),
                                SsaKind::GlobalStripElementPtr {
                                    param,
                                    kind: GlobalStripKind::ColStrip,
                                },
                            );
                        } else if op.get_num_results() == 1 {
                            update(&mut env, op.get_result(0), SsaKind::DataValue);
                        }
                    }
                    _ => {
                        if op.get_num_results() == 1 {
                            update(&mut env, op.get_result(0), SsaKind::DataValue);
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    // 4b. Collect thread-coordinate branch guards (`GuardExpr`) and async `mbarrier`
    //     token chains (`StageProduced` -> `StageReady`) from `mir.cond_br`.
    let mut branch_guards: Vec<(usize, usize, GuardExpr)> = Vec::new();
    let mut arrived_expect_tx_bars: BTreeSet<usize> = BTreeSet::new();
    let mut async_bar_for_shared: HashMap<usize, usize> = HashMap::new();
    let mut mbarrier_wait_edges: Vec<(usize, usize, usize)> = Vec::new();
    for (w_blk, &b_ptr) in blocks.iter().enumerate() {
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            let opid = Operation::get_opid(op_ptr, ctx).to_string();
            let op = op_ptr.deref(ctx);
            if matches!(
                opid.as_str(),
                "nvvm.mbarrier_arrive_expect_tx_shared" | "nvvm.mbarrier_arrive_expect_tx_cluster"
            ) && op.get_num_operands() >= 1
                && let Some(SsaKind::MbarrierPtr { bar_id }) = env.get_kind(&op.get_operand(0))
            {
                arrived_expect_tx_bars.insert(bar_id);
            } else if opid.starts_with("nvvm.cp_async_bulk_tensor_g2s")
                && op.get_num_operands() >= 2
                && let (
                    Some(SsaKind::SharedBasePtr { shared_id, .. }),
                    Some(SsaKind::MbarrierPtr { bar_id }),
                ) = (
                    env.get_kind(&op.get_operand(0)),
                    env.get_kind(&op.get_operand(1)),
                )
            {
                async_bar_for_shared.insert(shared_id, bar_id);
            } else if opid == "mir.cond_br"
                && op.get_num_operands() >= 1
                && op.get_num_successors() == 2
            {
                let cond_kind = env.get_kind(&op.get_operand(0));
                let true_succ = op.get_successor(0);
                let false_succ = op.get_successor(1);
                if let (Some(t_idx), Some(f_idx)) = (
                    blocks.iter().position(|&b| b == true_succ),
                    blocks.iter().position(|&b| b == false_succ),
                ) {
                    match cond_kind {
                        Some(SsaKind::Guard(g)) => {
                            let is_loop_test = matches!(
                                &g,
                                GuardExpr::Lt(a, _) | GuardExpr::Ge(a, _) if a.loop_bound() > 1
                            );
                            if !is_loop_test {
                                if !can_reach(f_idx, t_idx, &succs, num_blocks) {
                                    branch_guards.push((t_idx, f_idx, g.clone()));
                                }
                                if !can_reach(t_idx, f_idx, &succs, num_blocks) {
                                    branch_guards.push((f_idx, t_idx, g.negate()));
                                }
                            }
                        }
                        Some(SsaKind::MbarrierReady { bar_id }) => {
                            mbarrier_wait_edges.push((w_blk, t_idx, bar_id));
                        }
                        Some(SsaKind::MbarrierNotReady { bar_id }) => {
                            mbarrier_wait_edges.push((w_blk, f_idx, bar_id));
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    let block_guard_expr = |b_idx: usize| -> GuardExpr {
        let mut guard = GuardExpr::True;
        for (in_dst, out_dst, g) in &branch_guards {
            if !can_reach_without(0, b_idx, *in_dst, &succs, num_blocks)
                && can_reach_without(*in_dst, b_idx, *out_dst, &succs, num_blocks)
            {
                guard = guard.and(g.clone());
            }
        }
        guard
    };
    let is_stage_ready = |b_idx: usize, bar_id: usize| -> bool {
        arrived_expect_tx_bars.contains(&bar_id)
            && mbarrier_wait_edges.iter().any(|&(w_blk, ready_dst, b_id)| {
                b_id == bar_id
                    && b_idx != w_blk
                    && !can_reach_without(0, b_idx, ready_dst, &succs, num_blocks)
                    && can_reach(ready_dst, b_idx, &succs, num_blocks)
            })
    };
    let mut effective_num_barriers = barrier_blocks.len();
    let mut final_barriers_uniform = barriers_uniform;
    for &(_w_blk, ready_dst, bar_id) in &mbarrier_wait_edges {
        if arrived_expect_tx_bars.contains(&bar_id) {
            let uniform_wait = return_blocks
                .iter()
                .all(|&ret_blk| !can_reach_without(0, ret_blk, ready_dst, &succs, num_blocks));
            if uniform_wait {
                if effective_num_barriers == 0 {
                    final_barriers_uniform = true;
                }
                effective_num_barriers += 1;
            } else {
                final_barriers_uniform = false;
            }
        }
    }

    // 5. Collect 2D global strip accesses, phased shared-memory accesses, and
    //    proven `mir.assert` nodes (rejecting any verified global slice or
    //    shared-memory `mir.ptr_offset` that did not resolve to a recognized
    //    coordinate expression).
    let mut global_strip_accesses = Vec::new();
    let mut shared_accesses = Vec::new();
    let mut proven_assert_ops = Vec::new();
    for (b_idx, &b_ptr) in blocks.iter().enumerate() {
        let dom_guard = block_guard_expr(b_idx);
        let leader_only = dom_guard.has_zero_leader_check();
        let active_threads = dom_guard.active_threads_bound(u64::from(spec.block_size));
        let residual_guard = dom_guard.custom_residual();
        let mut seen_barriers_in_block = 0usize;
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            let opid = Operation::get_opid(op_ptr, ctx).to_string();
            if opid == "nvvm.barrier0" {
                seen_barriers_in_block += 1;
                continue;
            }
            let base_phase = block_phase(b_idx, seen_barriers_in_block);
            let op = op_ptr.deref(ctx);
            if opid.starts_with("nvvm.cp_async_bulk_tensor_g2s") {
                if op.get_num_operands() >= 2
                    && let (
                        Some(SsaKind::SharedBasePtr { shared_id, size }),
                        Some(SsaKind::MbarrierPtr { .. }),
                    ) = (
                        env.get_kind(&op.get_operand(0)),
                        env.get_kind(&op.get_operand(1)),
                    )
                {
                    let shared_name = shared_raw_list
                        .get(shared_id)
                        .map(|(n, _)| n.clone())
                        .unwrap_or_else(|| format!("SHARED_{shared_id}"));
                    shared_accesses.push(PhasedSharedAccess {
                        kind: AccessKind::Write,
                        shared_id,
                        shared_name,
                        buffer_len: size,
                        phase: base_phase,
                        pattern: TileAccessPatternKind::LinearScan,
                        loop_bound: size,
                        leader_only,
                        active_threads,
                        guard: GuardExpr::True,
                    });
                }
                continue;
            }
            match opid.as_str() {
                "mir.ptr_offset" => {
                    if let Some(SsaKind::SharedBasePtr { .. }) = env.get_kind(&op.get_operand(0))
                        && !matches!(
                            env.get_kind(&op.get_result(0)),
                            Some(SsaKind::SharedElementPtr { .. })
                        )
                    {
                        return Err(reject(
                            "mir.ptr_offset",
                            "shared memory pointer offset must use a recognized 2D tile coordinate",
                        ));
                    }
                    if let Some(SsaKind::SlicePtr(param)) = env.get_kind(&op.get_operand(0))
                        && spec.no_oob_idx.contains(&param)
                        && !matches!(
                            env.get_kind(&op.get_result(0)),
                            Some(SsaKind::GlobalStripElementPtr { .. })
                        )
                    {
                        return Err(reject(
                            "mir.ptr_offset",
                            "2D global slice pointer offset must use a row-strip (`start + col`) or col-strip (`start + row * stride`) index",
                        ));
                    }
                }
                "mir.assert" => {
                    if op.get_num_operands() == 1 {
                        match env.get_kind(&op.get_operand(0)) {
                            Some(SsaKind::Guard(GuardExpr::Lt(_, _)))
                                if !spec.no_oob_shared.is_empty() =>
                            {
                                proven_assert_ops.push(op_ptr);
                            }
                            Some(SsaKind::IndexLtLen {
                                guard_param: checked_param,
                            }) if spec.no_oob_idx.contains(&checked_param) => {
                                proven_assert_ops.push(op_ptr);
                            }
                            _ => {}
                        }
                    }
                }
                "mir.load" => match env.get_kind(&op.get_operand(0)) {
                    Some(SsaKind::SharedElementPtr {
                        shared_id,
                        size,
                        addr,
                    }) => {
                        let mut phase = base_phase;
                        if let Some(&bar_id) = async_bar_for_shared.get(&shared_id)
                            && is_stage_ready(b_idx, bar_id)
                        {
                            phase += 1;
                        }
                        let (pattern, loop_bound) = addr.classify_pattern(spec.block_size);
                        let guard = if matches!(
                            pattern,
                            TileAccessPatternKind::Expr(_) | TileAccessPatternKind::ColScan { .. }
                        ) && active_threads < u64::from(spec.block_size)
                        {
                            dom_guard.clone()
                        } else {
                            residual_guard.clone()
                        };
                        let shared_name = shared_raw_list
                            .get(shared_id)
                            .map(|(n, _)| n.clone())
                            .unwrap_or_else(|| format!("SHARED_{shared_id}"));
                        shared_accesses.push(PhasedSharedAccess {
                            kind: AccessKind::Read,
                            shared_id,
                            shared_name,
                            buffer_len: size,
                            phase,
                            pattern,
                            loop_bound,
                            leader_only,
                            active_threads,
                            guard,
                        });
                    }
                    Some(SsaKind::GlobalStripElementPtr { param, kind }) => {
                        global_strip_accesses.push(GlobalStripAccess2D { param, kind });
                    }
                    _ => {}
                },
                "mir.store" => {
                    if let Some(SsaKind::SharedElementPtr {
                        shared_id,
                        size,
                        addr,
                    }) = env.get_kind(&op.get_operand(0))
                    {
                        let mut phase = base_phase;
                        if let Some(&bar_id) = async_bar_for_shared.get(&shared_id)
                            && is_stage_ready(b_idx, bar_id)
                        {
                            phase += 1;
                        }
                        let (pattern, loop_bound) = addr.classify_pattern(spec.block_size);
                        let guard = if matches!(
                            pattern,
                            TileAccessPatternKind::Expr(_) | TileAccessPatternKind::ColScan { .. }
                        ) && active_threads < u64::from(spec.block_size)
                        {
                            dom_guard.clone()
                        } else {
                            residual_guard.clone()
                        };
                        let shared_name = shared_raw_list
                            .get(shared_id)
                            .map(|(n, _)| n.clone())
                            .unwrap_or_else(|| format!("SHARED_{shared_id}"));
                        shared_accesses.push(PhasedSharedAccess {
                            kind: AccessKind::Write,
                            shared_id,
                            shared_name,
                            buffer_len: size,
                            phase,
                            pattern,
                            loop_bound,
                            leader_only,
                            active_threads,
                            guard,
                        });
                    }
                }
                _ => {}
            }
        }
    }
    shared_accesses.sort_by_key(|a| (a.shared_id, a.phase, a.kind));

    Ok(TranslatedFunction::TiledKernel2D {
        spec,
        num_barriers: effective_num_barriers,
        barriers_uniform: final_barriers_uniform,
        global_strip_accesses,
        shared_accesses,
        proven_assert_ops,
    })
}
