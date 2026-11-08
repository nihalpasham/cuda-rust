# Verified Kernels: From Safe(ish) SIMT to Safe SIMT

On a CPU, **Safe Rust** means your program has zero undefined behavior:
- No out-of-bounds memory accesses
- No data races between threads
- No dangling pointers

Notice what Safe Rust does *not* mean: it does not prove **functional correctness** (that your math formula is right). Writing `a[i] - b[i]` when you meant `a[i] + b[i]` is a logic bug, but it is still 100% memory-safe Rust.

On a GPU, thousands of threads run the same kernel function at once, and the Rust borrow checker cannot reason about thread coordinates or barriers:
- **What Rust types already cover**: [The Safety Model](the-safety-model.md) proves one-thread-per-element writes with disjoint slices, and [Bounds Checks](bounds-checks.md) hoists row and column checks with matrix views.
- **Where Rust types stop**: When 256 threads share an on-chip tile across barriers, the borrow checker cannot tell if two threads collide or if a barrier sits inside an `if` branch. Until now, those kernels needed `unsafe` blocks or unchecked indexing flags.
- **How we close the gap**: The `#[verify(...)]` attribute pairs Rust's type system with compile-time **Lean 4** proofs, turning "safe(ish) SIMT" into **Safe SIMT**.

```{figure} images/three-gpu-hazards.svg
:alt: Three GPU safety hazards proved at compile time by the verify attribute
:align: center
:width: 100%

The three GPU safety bugs caught at compile time: out-of-bounds indexing, same-phase shared-memory data races, and divergent barriers.
```

---

## 1. The Core Idea: Specify and Verify

You declare the safety properties you want on the kernel, and the compiler proves them before generating PTX.

We do not prove functional correctness or floating-point math formulas. Instead, we prove the exact memory and concurrency rules needed to make the kernel safe and strip out redundant bounds checks:

| Property You Write  | What Lean 4 Proves at Compile Time                        | What Is Not Claimed                       |
|:--------------------|:----------------------------------------------------------|:------------------------------------------|
| `no_oob_idx`        | Every buffer read and write index stays below its length  | Values stored at those indices (`NaN`)    |
| `no_data_race`      | No two threads conflict on shared memory in the same step | Shared-memory bank conflicts (speed only) |
| `disjoint_writes`   | Two different threads never write to the same output slot | Which formula is written (`a - b` vs `+`) |
| `writes_all_elems`  | Every output slot is written by a launched thread         | Whether the written value is right        |
| `uniform_barriers`  | Every thread in the block reaches every barrier           | Waiting across different thread blocks    |

---

## 2. How It Works

Verification splits the work between two places so GPU threads pay zero runtime cost:
1. **At launch time on the CPU**: The host launcher checks buffer sizes and block dimensions once before launching the kernel.
2. **At compile time in Lean 4**: The compiler pauses at `PostMem2Reg` (after promoting variables to SSA registers in `dialect-mir`), hands the kernel's indexing structure to Lean 4, and deletes the proven bounds checks before emitting PTX.

```{figure} images/specify-and-verify-pipeline.svg
:alt: Specify and Verify pipeline
:align: center
:width: 100%

The compiler pauses in memory at `PostMem2Reg`, proves the kernel's obligations in Lean 4, erases the proven bounds checks, and resumes lowering to PTX.
```

### Compile Time vs. Runtime

| Piece                 | Compile Time (Lean 4)                       | Runtime (Host CPU)                  | Runtime (GPU Threads)     |
|:----------------------|:--------------------------------------------|:------------------------------------|:--------------------------|
| Verification clauses  | Proves bounds, race freedom, and barriers   | Nothing                             | Bounds checks removed     |
| Block dimensions      | Taken as a given fact; stamped into PTX     | Checked once when preparing launch  | Hardware enforces block   |
| Length requirements   | Taken as a given assumption in the proof    | Checked once right before launch    | Nothing (already checked) |

### We Summarize the Kernel Instead of Translating Every Line

The Lean exporter has two modes:
- **Scalar mode**: Translates a pure integer function line-by-line into fixed-width bitvector math.
- **Kernel mode**: Walks the kernel's control-flow graph to extract a **kernel summary** (which buffers are touched, under what `if` guards, and in which barrier phases) and emits one theorem per verification clause. Floating-point additions and multiplies are ignored because they cannot change array indices.

How one `no_oob_idx` clause maps to the right reusable proof in Lean:
- **1D global slice** (`a[idx]`): proved from the `if` guard and the host length check.
- **2D global row or column strip**: proved from the strip start check and loop bound.
- **Shared memory tile**: proved from the 2D thread coordinates (`tx < 16`, `ty < 16`) and loop bound (`i < 16`).

### Trusted Computing Base

Lean 4's kernel checks every proof obligation at `PostMem2Reg` and rejects any proof that uses `sorryAx` or custom axioms. Everything outside that proof check is part of the **Trusted Computing Base (TCB)**:
- **MIR import and SSA promotion (`mir-importer`, `mem2reg`)**: Translating Rust into `dialect-mir` and promoting stack locals to `PostMem2Reg` SSA registers.
- **The Lean translator (`lean-exporter`)**: Extracting the kernel summary and proof obligations from `PostMem2Reg` `dialect-mir` (fails closed on unsupported ops, unguarded accesses, or unknown attributes).
- **Downstream lowering (`mir-transforms`, `mir-lower`, LLVM, `ptxas`)**: Erasing proven `mir.assert` checks, unrolling loops, and lowering `dialect-mir` to PTX after verification succeeds.
- **Host launch checks and CUDA runtime**: Enforcing `#[launch_contract]` block dimensions and buffer length relations before the kernel runs.

---

## 3. Reading Verification Errors

When a kernel has a bug, compilation stops before PTX is generated:
- **Missing guards** (such as reading a slice outside an `if` check, or using `<=` instead of `<`) are caught immediately during the MIR walk.
- **Index math or barrier bugs** (such as forgetting a host length check, indexing `a[idx + 1]`, or forgetting a barrier between a shared write and read) fail in Lean 4's arithmetic solver, which prints a counterexample:

```text
error: omega could not prove the goal:
a possible counterexample may satisfy the constraints
  b - c ≥ 1
  a - b ≥ 0
  a - c ≤ 1
where
  a := ↑(lengths ⟨0, ...⟩)                 -- length of input a
  b := ↑(lengths ⟨2, ...⟩)                 -- length of output c
  c := ↑(Mir.globalIndex 256 block lane)   -- thread index
```

How to read that error in three quick steps:
1. **Read the `where` legend at the bottom first**: Lean renames expressions to `a`, `b`, and `c`. Here `a` is the length of `a`, `b` is the length of `c`, and `c` is the thread index.
2. **Plug those names into the inequalities**:
   - `b - c >= 1` means the thread passed the `if` check (`idx < c.len()`).
   - `a - b >= 0` means the host check passed (`a.len() >= c.len()`).
   - `a - c <= 1` means `idx + 1 >= a.len()`, so `a[idx + 1]` walks off the end of `a`!
3. **Test with simple numbers**: If both slices have length 10, thread 9 passes `9 < 10` and then reads `a[9 + 1] = a[10]`, which is out of bounds.

---

## 4. Walkthroughs

### Example 1: 1D Vector Addition

In `verified_vecadd`, each thread checks `c.get_mut(idx)` (`idx < c.len()`) and then reads `a[idx_raw]` and `b[idx_raw]`:

```rust
#[launch_contract(
    domain = 1,
    block = (256, 1, 1),
    requires = (a.len() >= c.len(), b.len() >= c.len()),
)]
#[verify(
    no_oob_idx(a, b, c),
    disjoint_writes(c),
    writes_all_elems(c),
)]
#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let idx_raw = idx.get();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[idx_raw] + b[idx_raw];
    }
}
```

What happens when you build:
- The host contract guarantees `a` and `b` are at least as long as `c`.
- Lean 4 proves that any thread inside `if let Some(c_elem)` is also in bounds for `a` and `b`.
- The compiler deletes both `a[idx_raw]` and `b[idx_raw]` bounds checks before PTX lowering.

You can inspect the MIR input and generated Lean theorems at any time:

```console
$ cargo oxide lean-export verified_vecadd
```

---

### Example 2: 2D Tiled Matrix Multiply (Shared Memory + Barriers)

In a 2D tiled matrix multiply (`verified_gemm_views`), a 16x16 block of 256 threads cooperates in two phases separated by a barrier:
- **Phase 0 (Stage)**: Each thread `(tx, ty)` writes one float into shared memory at its own slot (`ty * 16 + tx`).
- **Barrier**: All 256 threads wait at `sync_threads()` until both shared tiles are full.
- **Phase 1 (Scan)**: Each thread loops 16 times (`i = 0..16`), reading across row `ty` of tile A and down column `tx` of tile B.

```{figure} images/phased-shared-memory.svg
:alt: Phased shared memory verification across sync_threads
:align: center
:width: 100%

Phase 0 writes one unique slot per thread. After the barrier wall, Phase 1 only reads. Keeping writes and cross-thread reads in separate phases prevents data races.
```

Because Lean 4 proves `no_oob_idx`, `no_data_race`, and `uniform_barriers` at compile time:
- **Both `unsafe` blocks in `sgemm_tiled_views` are gone**: In `gemm_views`, Phase 0 staging and Phase 1 scanning each needed an `unsafe { ... }` block around `TILE_A` and `TILE_B`. Adding `#[verify(no_oob_idx(TILE_A, TILE_B), no_data_race(TILE_A, TILE_B), uniform_barriers)]` removes both `unsafe` blocks and elides all 4 shared-memory bounds checks.
- **Plain slice indexing is now safe too**: Even without matrix view types (`MatrixView32`), `sgemm_tiled_raw` is now 100% safe Rust and elides all 6 bounds checks (2 on global slices `a` and `b`, plus 4 on shared tiles `TILE_A` and `TILE_B`).

```rust
#[kernel(launch_context = lc)]
#[launch_bounds(256)]
#[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1),
    requires = (k >= 1, a.len() >= m * k, b.len() >= k * n, c.len() >= m * n))]
#[verify(
    no_oob_idx(a, b, TILE_A, TILE_B),
    no_data_race(TILE_A, TILE_B),
    uniform_barriers,
)]
pub fn sgemm_tiled_raw(
    m: u32,
    n: Uniform<u32>,
    k: u32,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    mut c: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>,
) {
    static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;
    // ...
    while tile < num_tiles {
        // ...
        // Phase 0 (Lean-verified bounds and disjoint per-thread slot):
        {
            TILE_A[smem_idx] = a_value;
            TILE_B[smem_idx] = b_value;
        }

        thread::sync_threads();

        // Phase 1 (Lean-verified bounds and barrier separation):
        {
            let mut i = 0usize;
            while i < TILE as usize {
                sum += TILE_A[ty as usize * 16 + i] * TILE_B[i * 16 + tx as usize];
                i += 1;
            }
        }

        thread::sync_threads();
        tile = tile.wrapping_add(1);
    }
    // ...
}
```

Inspect both verified kernels or run the benchmark with:

```console
$ cargo oxide lean-export verified_gemm_views
$ cargo oxide run verified_gemm_views
```

| Kernel Variant                   | Safety Status | Bounds Checks Removed by Lean 4  | Throughput (1024³) |
|:---------------------------------|:--------------|:---------------------------------|-------------------:|
| Naive with views                 | Safe          | None (uses matrix views)         | ~7,225 GFLOPS      |
| Tiled with views                 | Safe          | 4 shared-memory checks           | ~9,405 GFLOPS      |
| Tiled without views (`tiled_raw`)| **Now Safe!** | 2 global + 4 shared-memory checks| ~9,390 GFLOPS      |
