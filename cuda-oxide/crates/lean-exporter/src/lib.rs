/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Translates `dialect-mir` functions into Lean 4 code using a single typed SSA
//! environment ([`ssa::SsaEnv`]).
//!
//! ```text
//!   mir.func
//!     │
//!     ├─► 1. `spec.rs`      Read signature & optional `#[verify(...)]` attributes
//!     ├─► 2. `ssa.rs`       Typed SSA lookup table (`Value -> TypedSsa { lean_name, kind }`)
//!     ├─► 3. `translate.rs` Walk MIR instructions using `SsaEnv` -> `TranslatedFunction`
//!     ├─► 4. `codegen.rs`   Render `TranslatedFunction` into `.lean` code + theorems
//!     └─► 5. `runner.rs`    Run `lake env lean` & reject failed proofs or `sorryAx`
//! ```

pub mod codegen;
pub mod runner;
pub mod spec;
pub mod ssa;
pub mod translate;

pub use runner::verify_module_kernels;

use std::collections::HashSet;

use combine::{
    Parser, attempt,
    parser::char::{spaces, string},
    satisfy, skip_many,
};
use dialect_mir::ops::MirFuncOp;
use pliron::{
    attribute::AttributeDict,
    builtin::attributes::{GivenNamesAttr, IdentifierAttr},
    context::{Context, Ptr},
    irfmt::parsers::spaced,
    linked_list::ContainsLinkedList,
    operation::Operation,
    parsable::parse_from_str,
};

/// Unsupported and malformed input is rejected before any Lean is returned.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// Textual MIR failed to parse.
    #[error("cannot parse MIR: {0}")]
    Parse(String),
    /// An operation, attribute, or contract failed fail-closed validation.
    #[error("{scope}: {detail}")]
    Rejected {
        /// Diagnostic stage or operation scope.
        scope: String,
        /// Reason the input was rejected.
        detail: String,
    },
    /// Pliron structural or dominance verification failed.
    #[error("MIR structural verification failed: {0}")]
    Verification(String),
    /// The requested symbol was not present in the input MIR.
    #[error("function `{0}` was not found")]
    FunctionNotFound(String),
}

pub(crate) fn reject(scope: &str, detail: impl Into<String>) -> ExportError {
    ExportError::Rejected {
        scope: scope.into(),
        detail: detail.into(),
    }
}

/// Only presentation metadata is ignored; all other attributes must be known.
pub(crate) fn check_attributes(
    attrs: &AttributeDict,
    permitted: &[&str],
    scope: &str,
) -> Result<(), ExportError> {
    for (name, attr) in &attrs.0 {
        let name = name.to_string();
        if name == "builtin_given_names" {
            if attr.downcast_ref::<GivenNamesAttr>().is_none() {
                return Err(reject(scope, "malformed given-name metadata"));
            }
        } else if !permitted.contains(&name.as_str()) {
            return Err(reject(scope, format!("unsupported attribute `{name}`")));
        }
    }
    Ok(())
}

pub(crate) fn get_attribute<'a, T: pliron::attribute::Attribute>(
    attrs: &'a AttributeDict,
    name: &str,
    scope: &str,
) -> Result<&'a T, ExportError> {
    attrs
        .0
        .iter()
        .find(|(key, _)| key.to_string() == name)
        .and_then(|(_, value)| value.downcast_ref::<T>())
        .ok_or_else(|| reject(scope, format!("missing or malformed attribute `{name}`")))
}

pub(crate) fn symbol_name(ctx: &Context, op: Ptr<Operation>) -> Result<String, ExportError> {
    let operation = op.deref(ctx);
    Ok(
        get_attribute::<IdentifierAttr>(&operation.attributes, "builtin_sym_name", "function")?
            .as_ref()
            .to_string(),
    )
}

/// Parse an entire MIR file and export a function by its exact symbol name.
///
/// Accepts a standalone `mir.func` or a `builtin.module` of direct functions.
/// The selected function alone is exported. Trailing input is never ignored.
pub fn export_named_source(source: &str, name: &str) -> Result<String, ExportError> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    let headers = skip_many(attempt(
        spaces()
            .with(string("//"))
            .skip(skip_many(satisfy(|c| c != '\n'))),
    ));
    let root = parse_from_str(
        headers
            .with(spaced(Operation::top_level_parser()))
            .skip(combine::eof()),
        &mut ctx,
        source,
    )
    .map_err(|error| ExportError::Parse(error.to_string()))?;

    if MirFuncOp::wrap(&ctx, root).is_some() {
        if symbol_name(&ctx, root)? != name {
            return Err(ExportError::FunctionNotFound(name.into()));
        }
        return export_function(&ctx, root);
    }
    if Operation::get_opid(root, &ctx).to_string() != "builtin.module" {
        return Err(reject("input", "expected mir.func or builtin.module"));
    }
    let module = root.deref(&ctx);
    if module.num_regions() != 1
        || module.get_num_results() != 0
        || module.get_num_operands() != 0
        || module.get_num_successors() != 0
    {
        return Err(reject("module", "malformed module structure"));
    }
    check_attributes(&module.attributes, &["builtin_sym_name"], "module")?;
    get_attribute::<IdentifierAttr>(&module.attributes, "builtin_sym_name", "module")?;
    let region = module.get_region(0).deref(&ctx);
    let blocks: Vec<_> = region.iter(&ctx).collect();
    if blocks.len() != 1 {
        return Err(reject("module", "expected exactly one module block"));
    }
    let block = blocks[0].deref(&ctx);
    if block.get_num_arguments() != 0 {
        return Err(reject("module", "module block cannot have arguments"));
    }
    check_attributes(&block.attributes, &[], "module block")?;
    let mut selected = None;
    let mut symbols = HashSet::new();
    for child in block.iter(&ctx) {
        if MirFuncOp::wrap(&ctx, child).is_none() {
            return Err(reject(
                "module",
                "only direct mir.func children are supported",
            ));
        }
        let symbol = symbol_name(&ctx, child)?;
        if !symbols.insert(symbol.clone()) {
            return Err(reject(
                "module",
                format!("duplicate function symbol `{symbol}`"),
            ));
        }
        if symbol == name {
            selected = Some(child);
        }
    }
    export_function(
        &ctx,
        selected.ok_or_else(|| ExportError::FunctionNotFound(name.into()))?,
    )
}

/// Translate a `mir.func` operation into Lean 4 source code using the unified
/// [`translate::translate_function`] and [`codegen::render_lean_module`] pipeline.
pub fn export_function(ctx: &Context, function: Ptr<Operation>) -> Result<String, ExportError> {
    if MirFuncOp::wrap(ctx, function).is_none() {
        return Err(reject("function", "expected mir.func"));
    }
    let translated = translate::translate_function(ctx, function)?;
    codegen::render_lean_module(&translated)
}
