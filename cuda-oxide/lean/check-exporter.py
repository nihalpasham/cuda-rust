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
SGEMM_TILED_FIXTURE = REPO / "crates/lean-exporter/fixtures/sgemm_tiled.mir"
WARP_REDUCE_FIXTURE = REPO / "crates/lean-exporter/fixtures/warp_reduce.mir"
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

        # Mutate vecadd MIR with an off-by-one neighbor access `a[idx + 1]` under `idx < c.len()`:
        off_by_one_mir = temporary / "vecadd_off_by_one.mir"
        off_by_one_mir.write_text(
            vecadd_source.replace(
                "a_elem = mir.ptr_offset (a_ptr, idx)",
                "one = mir.constant () [] [value: builtin.integer <1: ui64>]: <() -> (builtin.integer ui64)>;\n"
                "    idx_plus_one = mir.add (idx, one) [] []: <(builtin.integer ui64, builtin.integer ui64) -> (builtin.integer ui64)>;\n"
                "    a_elem = mir.ptr_offset (a_ptr, idx_plus_one)",
                1,
            )
        )
        off_by_one_lean = require_export(off_by_one_mir, "vecadd")
        off_by_one_file = temporary / "VecaddOffByOne.lean"
        off_by_one_file.write_text(
            "\n".join(line for line in off_by_one_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_oob = run_lake("env", "lean", str(off_by_one_file))
        if res_oob.returncode == 0 or "error: omega could not prove the goal:" not in res_oob.stdout:
            raise RuntimeError(f"Expected off-by-one `a[idx + 1]` access to fail in Lean:\n{res_oob.stdout}")
        print("REJECTED as expected: off-by-one `a[idx + 1]` access fails the `no_oob_idx(a)` proof.", flush=True)

        # Mutate vecadd MIR with an inclusive guard `mir.le (idx, c_len)` (`idx <= c.len()`):
        inclusive_mir = temporary / "vecadd_inclusive.mir"
        inclusive_mir.write_text(vecadd_source.replace("mir.lt (idx, c_len)", "mir.le (idx, c_len)", 1))
        rejected_inc = export(inclusive_mir, "vecadd")
        if rejected_inc.returncode == 0 or not rejected_inc.stderr:
            raise RuntimeError("Expected inclusive `idx <= c.len()` guard to be rejected by exporter")
        print("REJECTED as expected: inclusive `idx <= c.len()` guard is rejected before Lean.", flush=True)

        # Verified 2D phased shared-memory SGEMM checks
        sgemm_source = SGEMM_TILED_FIXTURE.read_text()
        sgemm_lean = require_export(SGEMM_TILED_FIXTURE, "sgemm_tiled_views")
        sgemm_valid = temporary / "SgemmValid.lean"
        sgemm_valid.write_text(sgemm_lean)
        require_success("env", "lean", str(sgemm_valid))
        print("PASS: exported 2D sgemm_tiled_views MIR proves no_oob_idx, no_data_race, and uniform_barriers.", flush=True)

        # Mutate sgemm_tiled_views MIR by removing the first `nvvm.barrier0` (`sync_threads()`):
        race_mir = temporary / "sgemm_missing_barrier.mir"
        race_mir.write_text(sgemm_source.replace("    nvvm.barrier0 () [] []: <() -> ()>;\n", "", 1))
        race_lean = require_export(race_mir, "sgemm_tiled_views")
        race_file = temporary / "SgemmRace.lean"
        race_file.write_text(
            "\n".join(line for line in race_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_race = run_lake("env", "lean", str(race_file))
        if res_race.returncode == 0 or "error: omega could not prove the goal:" not in res_race.stdout:
            raise RuntimeError(f"Expected missing `sync_threads()` to fail NoDataRace in Lean:\n{res_race.stdout}")
        print("REJECTED as expected: missing `sync_threads()` between write and read fails `no_data_race`.", flush=True)

        # Verified 2D sgemm_tiled_raw global strip checks (RowStrip on `a`, ColStrip on `b`)
        sgemm_raw_lean = require_export(SGEMM_TILED_FIXTURE, "sgemm_tiled_raw")
        sgemm_raw_valid = temporary / "SgemmRawValid.lean"
        sgemm_raw_valid.write_text(sgemm_raw_lean)
        require_success("env", "lean", str(sgemm_raw_valid))
        print("PASS: exported 2D sgemm_tiled_raw MIR proves global RowStrip/ColStrip and shared-memory obligations.", flush=True)

        # Mutate sgemm_tiled_raw MIR to index `a` with an unguarded/non-strip offset `a_len`:
        bad_strip_mir = temporary / "sgemm_bad_strip.mir"
        bad_strip_mir.write_text(
            sgemm_source.replace(
                "a_elem = mir.ptr_offset (a_ptr, a_idx)",
                "a_elem = mir.ptr_offset (a_ptr, a_len)",
                1,
            )
        )
        rejected_strip = export(bad_strip_mir, "sgemm_tiled_raw")
        if rejected_strip.returncode == 0 or "2D global slice pointer offset must use a row-strip" not in rejected_strip.stderr:
            raise RuntimeError(f"Expected non-strip 2D global slice access to be rejected:\n{rejected_strip.stderr}")
        print("REJECTED as expected: non-strip 2D global slice access is rejected before Lean.", flush=True)

        # Stage 3: Verified warp & 2-phase block reductions
        warp_source = WARP_REDUCE_FIXTURE.read_text()
        warp_lean = require_export(WARP_REDUCE_FIXTURE, "warp_reduce_sum")
        warp_valid = temporary / "WarpReduceValid.lean"
        warp_valid.write_text(warp_lean)
        require_success("env", "lean", str(warp_valid))
        print("PASS: exported warp_reduce_sum MIR proves 1D warp butterfly reduction obligations.", flush=True)

        block_lean = require_export(WARP_REDUCE_FIXTURE, "block_reduce_sum")
        block_valid = temporary / "BlockReduceValid.lean"
        block_valid.write_text(block_lean)
        require_success("env", "lean", str(block_valid))
        print("PASS: exported block_reduce_sum MIR proves leader-guarded WARP_SUMS[tid / 32] and Phase 1 scan.", flush=True)

        # Mutate block_reduce_sum MIR by removing the `if lane == 0` guard:
        no_leader_mir = temporary / "block_reduce_no_leader.mir"
        no_leader_mir.write_text(
            warp_source.replace(
                "mir.cond_br (is_leader) [^leader_stage, ^barrier_sync] [builtin_operand_segment_sizes: builtin.operand_segment_sizes [1, 0, 0]]: <(builtin.integer i1) -> ()>",
                "mir.goto () [^leader_stage] []: <() -> ()>",
                1,
            )
        )
        no_leader_lean = require_export(no_leader_mir, "block_reduce_sum")
        no_leader_file = temporary / "BlockReduceNoLeader.lean"
        no_leader_file.write_text(
            "\n".join(line for line in no_leader_lean.splitlines() if not line.startswith("#print axioms"))
        )
        res_no_leader = run_lake("env", "lean", str(no_leader_file))
        if res_no_leader.returncode == 0 or "error: omega could not prove the goal:" not in res_no_leader.stdout:
            raise RuntimeError(f"Expected missing `lane == 0` guard to fail NoDataRace in Lean:\n{res_no_leader.stdout}")
        print("REJECTED as expected: missing `lane == 0` guard before `WARP_SUMS[tid / 32]` fails `no_data_race`.", flush=True)

    print("Exporter checks passed for scalar, 1D vecadd, 2D SGEMM, and Stage 3 warp/block reduction fixtures.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
