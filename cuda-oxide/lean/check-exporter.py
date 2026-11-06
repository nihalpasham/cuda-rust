#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the MIR exporter and Lean together on scalar and vecadd MIR fixtures."""

from pathlib import Path
import subprocess
import sys
import tempfile

from check import ROOT, require_success, run_lake

REPO = ROOT.parent
SCALAR_FIXTURE = REPO / "crates/lean-exporter/fixtures/arithmetic.mir"
VECADD_FIXTURE = REPO / "crates/lean-exporter/fixtures/vecadd.mir"
PROOFS = """
namespace CudaOxide.Generated

theorem index_formula (inputs : Fin 3 → BitVec 64) :
    evaluate inputs =
      inputs ⟨0, by decide⟩ * inputs ⟨1, by decide⟩ + inputs ⟨2, by decide⟩ := by
  rfl

example : evaluate (fun i =>
    if i.val = 0 then 3 else if i.val = 1 then 256 else 255) =
    (1023 : BitVec 64) := by decide

-- The exported arithmetic must retain the source operation's wraparound.
example : evaluate (fun i =>
    if i.val = 0 then 18446744073709551615 else 1) =
    (0 : BitVec 64) := by decide

#print axioms index_formula
end CudaOxide.Generated
"""


def export(path, fn_name="global_index"):
    return subprocess.run(
        ["cargo", "run", "--quiet", "--package", "lean-exporter", "--",
         str(path), fn_name],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def require_export(path, fn_name="global_index"):
    result = export(path, fn_name)
    if result.returncode != 0:
        raise RuntimeError(f"Exporter failed for {path} ({fn_name}):\n{result.stderr}")
    return result.stdout


def main():
    require_success("build")
    source = SCALAR_FIXTURE.read_text()
    if source.count("mir.mul") != 1:
        raise RuntimeError("The arithmetic fixture must contain exactly one mir.mul")
    with tempfile.TemporaryDirectory(prefix="cuda-oxide-lean-export-") as directory:
        temporary = Path(directory)
        generated = require_export(SCALAR_FIXTURE)
        valid = temporary / "Valid.lean"
        valid.write_text(generated + PROOFS)
        require_success("env", "lean", str(valid))
        print("PASS: generated Lean proves the MIR arithmetic formula and wraparound.", flush=True)

        mutated_mir = temporary / "mutated.mir"
        mutated_mir.write_text(source.replace("mir.mul", "mir.add", 1))
        mutated = require_export(mutated_mir)
        if mutated == generated:
            raise RuntimeError("Changing a MIR operation did not change the generated Lean")
        invalid = temporary / "Mutated.lean"
        invalid.write_text(mutated + PROOFS.replace("#print axioms index_formula\n", ""))
        result = run_lake("env", "lean", str(invalid))
        if result.returncode == 0 or "Tactic `rfl` failed" not in result.stdout:
            raise RuntimeError(f"Expected the arithmetic formula proof to fail:\n{result.stdout}")
        print("REJECTED as expected: changed MIR no longer proves the original formula.", flush=True)

        unsupported_mir = temporary / "unsupported.mir"
        unsupported_mir.write_text(source.replace("mir.mul", "mir.div", 1))
        rejected = export(unsupported_mir)
        if rejected.returncode <= 0 or rejected.stdout or "mir.div" not in rejected.stderr:
            raise RuntimeError(f"Expected explicit rejection without Lean output:\n{rejected.stderr}")
        print("REJECTED as expected: unsupported MIR operation emits no Lean.", flush=True)

        # Verified vecadd kernel MIR checks
        vecadd_source = VECADD_FIXTURE.read_text()
        vecadd_lean = require_export(VECADD_FIXTURE, "vecadd")
        vecadd_valid = temporary / "VecaddValid.lean"
        vecadd_valid.write_text(vecadd_lean)
        require_success("env", "lean", str(vecadd_valid))
        print("PASS: exported vecadd MIR proves no_oob_idx, disjoint_writes, and writes_all_elems.", flush=True)

        # Mutate vecadd MIR by dropping the `0>=2` (`a.len() >= c.len()`) precondition:
        missing_req_mir = temporary / "vecadd_missing_req.mir"
        missing_req_mir.write_text(vecadd_source.replace("0>=2,1>=2", "1>=2", 1))
        missing_req_lean = require_export(missing_req_mir, "vecadd")
        missing_req_file = temporary / "VecaddMissingReq.lean"
        missing_req_file.write_text(
            "\n".join(line for line in missing_req_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_missing = run_lake("env", "lean", str(missing_req_file))
        if res_missing.returncode == 0 or "error: omega could not prove the goal:" not in res_missing.stdout:
            raise RuntimeError(f"Expected missing `a.len() >= c.len()` precondition to fail in Lean:\n{res_missing.stdout}")
        print("REJECTED as expected: removing `a.len() >= c.len()` breaks the `no_oob_idx(a)` proof.", flush=True)

    print("Exporter checks passed for both scalar and verified vecadd MIR fixtures.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
