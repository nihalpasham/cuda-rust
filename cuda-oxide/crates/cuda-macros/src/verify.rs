/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Parsing, parameter validation, and compiler-marker injection for the
//! `#[verify(...)]` formal-verification attribute.

use crate::common::attr_path_ends_with;
use crate::cuda_module::model::{CudaModuleParam, CudaModuleParamMarshal};
use crate::launch_attrs::{requires_params_from_inputs, standalone_requires_params};
use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Expr, FnArg, Ident, Item, ItemFn, Stmt, Token, Type,
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote,
    punctuated::Punctuated,
    visit_mut::{self, VisitMut},
};

/// Extract `(lhs_param_index, rhs_param_index)` pairs for `requires` relations
/// of the form `lhs.len() >= rhs.len()` or `rhs.len() <= lhs.len()`.
pub(crate) fn extract_requires_len_ge_pairs(
    relations: &[Expr],
    params: &[CudaModuleParam],
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for relation in relations {
        let Expr::Binary(binary) = relation else {
            continue;
        };
        let (Some(left_idx), Some(right_idx)) = (
            slice_len_param_index(&binary.left, params),
            slice_len_param_index(&binary.right, params),
        ) else {
            continue;
        };
        match binary.op {
            syn::BinOp::Ge(_) => pairs.push((left_idx, right_idx)),
            syn::BinOp::Le(_) => pairs.push((right_idx, left_idx)),
            _ => {}
        }
    }
    pairs
}

fn slice_len_param_index(expr: &Expr, params: &[CudaModuleParam]) -> Option<usize> {
    let Expr::MethodCall(call) = expr else {
        return None;
    };
    if call.method != "len" || !call.args.is_empty() {
        return None;
    }
    let Expr::Path(path) = &*call.receiver else {
        return None;
    };
    let ident = path.path.get_ident()?;
    params.iter().position(|param| param.name == *ident)
}

/// Parsed clauses of `#[verify(...)]`.
#[derive(Clone, Default)]
pub(crate) struct VerifyArgs {
    /// Targets of `no_oob_idx(...)` (slice parameters or `SharedArray` statics).
    pub(crate) no_oob_idx: Vec<Ident>,
    /// Targets of `disjoint_writes(...)` (writable slice parameters).
    pub(crate) disjoint_writes: Vec<Ident>,
    /// Targets of `writes_all_elems(...)` (writable slice parameters).
    pub(crate) writes_all_elems: Vec<Ident>,
    /// Targets of `no_data_race(...)` / `no_data_races(...)` (`SharedArray` statics or writable slices).
    pub(crate) no_data_race: Vec<Ident>,
    /// Span of `uniform_barriers` when requested.
    pub(crate) uniform_barriers: Option<Ident>,
}

impl Parse for VerifyArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Err(syn::Error::new(
                input.span(),
                "verify requires at least one obligation: `no_oob_idx(...)`, `disjoint_writes(...)`, `writes_all_elems(...)`, `no_data_race(...)`, or `uniform_barriers`",
            ));
        }
        let mut args = VerifyArgs::default();
        while !input.is_empty() {
            let clause: Ident = input.parse()?;
            if clause == "uniform_barriers" {
                if input.peek(syn::token::Paren) {
                    let content;
                    syn::parenthesized!(content in input);
                    if !content.is_empty() {
                        return Err(syn::Error::new(
                            clause.span(),
                            "`uniform_barriers` takes no arguments",
                        ));
                    }
                }
                args.uniform_barriers = Some(clause);
                if !input.is_empty() {
                    input.parse::<Token![,]>()?;
                }
                continue;
            }
            let content;
            syn::parenthesized!(content in input);
            let params: Punctuated<Ident, Token![,]> = Punctuated::parse_terminated(&content)?;
            if params.is_empty() {
                return Err(syn::Error::new(
                    clause.span(),
                    format!("`{clause}(...)` requires at least one buffer or shared-array name"),
                ));
            }
            match clause.to_string().as_str() {
                "no_oob_idx" => args.no_oob_idx.extend(params),
                "disjoint_writes" => args.disjoint_writes.extend(params),
                "writes_all_elems" => args.writes_all_elems.extend(params),
                "no_data_race" | "no_data_races" => args.no_data_race.extend(params),
                _ => {
                    return Err(syn::Error::new(
                        clause.span(),
                        format!(
                            "unknown #[verify] obligation `{clause}`; expected `no_oob_idx`, `disjoint_writes`, `writes_all_elems`, `no_data_race`, or `uniform_barriers`"
                        ),
                    ));
                }
            }
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(args)
    }
}

/// Entry point for the `#[verify(...)]` attribute macro.
pub(crate) fn verify_entry(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as VerifyArgs);
    let mut input = parse_macro_input!(item as ItemFn);

    if let Some(params) = standalone_requires_params(&input)
        && let Err(error) = resolve_and_inject_verify_markers(&args, &params, &mut input)
    {
        return error.to_compile_error().into();
    }

    quote! { #input }.into()
}

/// Validate and build marker statements for any `#[verify(...)]` attributes
/// routed across generic kernel wrappers, and lift verified `SharedArray`
/// accesses so kernel authors do not need manual `unsafe` blocks.
pub(crate) fn validate_routed_verify_attrs(
    attrs: &[syn::Attribute],
    source_inputs: &syn::punctuated::Punctuated<FnArg, syn::token::Comma>,
    body_stmts: &mut [Stmt],
) -> syn::Result<Vec<Stmt>> {
    let mut markers = Vec::new();
    let Some(params) = requires_params_from_inputs(source_inputs) else {
        return Ok(markers);
    };
    let shared_statics = collect_shared_array_statics(body_stmts);
    for attr in attrs {
        if !attr_path_ends_with(attr, "verify") {
            continue;
        }
        let args = attr.parse_args::<VerifyArgs>()?;
        markers.extend(build_verify_marker_stmts(&args, &params, &shared_statics)?);
        rewrite_verified_shared_accesses(&args, &shared_statics, body_stmts);
    }
    Ok(markers)
}

fn resolve_and_inject_verify_markers(
    args: &VerifyArgs,
    params: &[CudaModuleParam],
    input: &mut ItemFn,
) -> syn::Result<()> {
    let shared_statics = collect_shared_array_statics(&input.block.stmts);
    let stmts = build_verify_marker_stmts(args, params, &shared_statics)?;
    rewrite_verified_shared_accesses(args, &shared_statics, &mut input.block.stmts);
    for (idx, stmt) in stmts.into_iter().enumerate() {
        input.block.stmts.insert(idx, stmt);
    }
    Ok(())
}

/// AST visitor that wraps index reads and writes on Lean-verified `SharedArray`
/// statics (`no_oob_idx` + `no_data_race` + `uniform_barriers`) in `unsafe` blocks
/// so the kernel body can be written in safe Rust with explicit phase scopes.
struct VerifiedSharedArrayRewriter {
    verified_shared: Vec<Ident>,
    in_unsafe: bool,
}

impl VerifiedSharedArrayRewriter {
    fn is_verified_shared_index(&self, expr: &Expr) -> bool {
        let Expr::Index(index_expr) = expr else {
            return false;
        };
        let Expr::Path(path_expr) = &*index_expr.expr else {
            return false;
        };
        path_expr
            .path
            .get_ident()
            .is_some_and(|id| self.verified_shared.iter().any(|v| v == id))
    }
}

impl VisitMut for VerifiedSharedArrayRewriter {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Expr::Unsafe(expr_unsafe) = expr {
            let prev = self.in_unsafe;
            self.in_unsafe = true;
            visit_mut::visit_expr_unsafe_mut(self, expr_unsafe);
            self.in_unsafe = prev;
            return;
        }

        if !self.in_unsafe {
            if let Expr::Assign(assign) = expr
                && self.is_verified_shared_index(&assign.left)
            {
                visit_mut::visit_expr_mut(self, &mut assign.right);
                if let Expr::Index(idx) = &mut *assign.left {
                    visit_mut::visit_expr_mut(self, &mut idx.index);
                }
                let inner = expr.clone();
                *expr = parse_quote! { unsafe { #inner } };
                return;
            }
            if self.is_verified_shared_index(expr) {
                if let Expr::Index(idx) = expr {
                    visit_mut::visit_expr_mut(self, &mut idx.index);
                }
                let inner = expr.clone();
                *expr = parse_quote! { unsafe { #inner } };
                return;
            }
        }

        visit_mut::visit_expr_mut(self, expr);
    }
}

fn rewrite_verified_shared_accesses(
    args: &VerifyArgs,
    shared_statics: &[Ident],
    stmts: &mut [Stmt],
) {
    if args.uniform_barriers.is_none() {
        return;
    }
    let verified_shared: Vec<Ident> = shared_statics
        .iter()
        .filter(|name| {
            args.no_oob_idx.iter().any(|n| n == *name)
                && args.no_data_race.iter().any(|n| n == *name)
        })
        .cloned()
        .collect();
    if verified_shared.is_empty() {
        return;
    }
    let mut rewriter = VerifiedSharedArrayRewriter {
        verified_shared,
        in_unsafe: false,
    };
    for stmt in stmts {
        rewriter.visit_stmt_mut(stmt);
    }
}

/// Collect in-body `static mut NAME: SharedArray<T, N>` declarations in source order.
fn collect_shared_array_statics(stmts: &[Stmt]) -> Vec<Ident> {
    let mut names = Vec::new();
    for stmt in stmts {
        if let Stmt::Item(Item::Static(item_static)) = stmt
            && is_shared_array_type(&item_static.ty)
        {
            names.push(item_static.ident.clone());
        }
    }
    names
}

fn is_shared_array_type(ty: &Type) -> bool {
    let Type::Path(type_path) = ty else {
        return false;
    };
    type_path
        .path
        .segments
        .last()
        .is_some_and(|seg| seg.ident == "SharedArray")
}

fn build_verify_marker_stmts(
    args: &VerifyArgs,
    params: &[CudaModuleParam],
    shared_statics: &[Ident],
) -> syn::Result<Vec<Stmt>> {
    let mut stmts = Vec::new();
    for ident in &args.no_oob_idx {
        if let Some(shared_idx) = shared_statics.iter().position(|name| name == ident) {
            stmts.push(parse_quote! {
                ::cuda_device::thread::__verify_no_oob_shared_config::<#shared_idx>();
            });
        } else {
            let index = resolve_verify_slice_param(ident, params, false, "no_oob_idx")?;
            stmts.push(parse_quote! {
                ::cuda_device::thread::__verify_no_oob_idx_config::<#index>();
            });
        }
    }
    for ident in &args.disjoint_writes {
        let index = resolve_verify_slice_param(ident, params, true, "disjoint_writes")?;
        stmts.push(parse_quote! {
            ::cuda_device::thread::__verify_disjoint_writes_config::<#index>();
        });
    }
    for ident in &args.writes_all_elems {
        let index = resolve_verify_slice_param(ident, params, true, "writes_all_elems")?;
        stmts.push(parse_quote! {
            ::cuda_device::thread::__verify_writes_all_elems_config::<#index>();
        });
    }
    for ident in &args.no_data_race {
        if let Some(shared_idx) = shared_statics.iter().position(|name| name == ident) {
            stmts.push(parse_quote! {
                ::cuda_device::thread::__verify_no_data_race_config::<#shared_idx>();
            });
        } else {
            let index = resolve_verify_slice_param(ident, params, true, "no_data_race")?;
            stmts.push(parse_quote! {
                ::cuda_device::thread::__verify_disjoint_writes_config::<#index>();
            });
        }
    }
    if args.uniform_barriers.is_some() {
        stmts.push(parse_quote! {
            ::cuda_device::thread::__verify_uniform_barriers_config();
        });
    }
    Ok(stmts)
}

fn resolve_verify_slice_param(
    ident: &Ident,
    params: &[CudaModuleParam],
    require_writable: bool,
    clause: &str,
) -> syn::Result<usize> {
    let Some((index, param)) = params
        .iter()
        .enumerate()
        .find(|(_, param)| param.name == *ident)
    else {
        return Err(syn::Error::new(
            ident.span(),
            format!("unknown kernel parameter `{ident}` in `{clause}(...)`"),
        ));
    };
    match param.marshal {
        CudaModuleParamMarshal::WritableDeviceBuffer { .. }
        | CudaModuleParamMarshal::RowWidthDeviceBuffer { .. } => Ok(index),
        CudaModuleParamMarshal::ReadOnlyDeviceBuffer { .. } if !require_writable => Ok(index),
        CudaModuleParamMarshal::ReadOnlyDeviceBuffer { .. } => Err(syn::Error::new(
            ident.span(),
            format!(
                "`{clause}({ident})` requires a writable buffer (`DisjointSlice` or `&mut [T]`), but `{ident}` is read-only"
            ),
        )),
        CudaModuleParamMarshal::Scalar => Err(syn::Error::new(
            ident.span(),
            format!(
                "`{clause}({ident})` requires a slice or `DisjointSlice` parameter, found scalar `{ident}`"
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_rejects_unknown_clause_readonly_writes_and_scalar_params() {
        let bad_clause = syn::parse_str::<VerifyArgs>("magic_clause(a)");
        assert!(
            bad_clause
                .err()
                .unwrap()
                .to_string()
                .contains("unknown #[verify] obligation `magic_clause`")
        );

        let func: ItemFn = parse_quote! {
            pub fn vecadd(a: &[f32], n: u32, mut c: DisjointSlice<f32>) {}
        };
        let params = standalone_requires_params(&func).expect("params");

        // `disjoint_writes(a)` on read-only slice `a: &[f32]` must be rejected.
        let readonly_write: VerifyArgs = syn::parse_str("disjoint_writes(a)").unwrap();
        let err = build_verify_marker_stmts(&readonly_write, &params, &[]).unwrap_err();
        assert!(err.to_string().contains("requires a writable buffer"));

        // `no_oob_idx(n)` on scalar `n: u32` must be rejected.
        let scalar_oob: VerifyArgs = syn::parse_str("no_oob_idx(n)").unwrap();
        let err = build_verify_marker_stmts(&scalar_oob, &params, &[]).unwrap_err();
        assert!(err.to_string().contains("found scalar `n`"));

        // `no_oob_idx(missing)` on unknown parameter must be rejected.
        let unknown_param: VerifyArgs = syn::parse_str("no_oob_idx(missing)").unwrap();
        let err = build_verify_marker_stmts(&unknown_param, &params, &[]).unwrap_err();
        assert!(
            err.to_string()
                .contains("unknown kernel parameter `missing`")
        );

        // `uniform_barriers(x)` with arguments must be rejected.
        let bad_uniform = syn::parse_str::<VerifyArgs>("uniform_barriers(a)");
        assert!(
            bad_uniform
                .err()
                .unwrap()
                .to_string()
                .contains("`uniform_barriers` takes no arguments")
        );

        // `no_oob_idx(TILE_A)` and `no_data_race(TILE_A)` on in-body SharedArray statics succeed.
        let tiled_fn: ItemFn = parse_quote! {
            pub fn sgemm(a: &[f32], mut c: DisjointSlice<f32>) {
                static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
            }
        };
        let shared = collect_shared_array_statics(&tiled_fn.block.stmts);
        assert_eq!(shared.len(), 1);
        let tiled_args: VerifyArgs =
            syn::parse_str("no_oob_idx(TILE_A), no_data_race(TILE_A), uniform_barriers").unwrap();
        let markers = build_verify_marker_stmts(&tiled_args, &params, &shared).unwrap();
        assert_eq!(markers.len(), 3);
    }
}
