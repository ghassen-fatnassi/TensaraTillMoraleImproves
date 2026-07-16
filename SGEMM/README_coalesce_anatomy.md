# Anatomy of `sgemm_global_mem_coalesce`: CUDA → PTX → SASS

This walks the memory-coalesced SGEMM kernel down through the three levels the
compiler produces:

1. **CUDA C++** — what you write.
2. **PTX** — NVIDIA's *virtual* ISA. Portable, SSA-ish, infinite virtual
   registers, target-independent-ish. Emitted by `nvcc`'s front-end (`-ptx`).
   `ptxas` then compiles this to SASS.
3. **SASS** — the *real* machine code for a specific GPU (here `sm_86`, your
   RTX 2050 / Ampere). Fixed register file, hardware scheduling, memory
   pipelines. This is what actually runs. Get it with
   `nvcc -arch=sm_86 -cubin x.cu && cuobjdump -sass x.cubin`.

> **Important correction about your third pasted block.** The block you labeled
> "SASS" is **not** SASS. It is **x86-64 host assembly** — the CPU-side launcher
> stub (`cudaLaunchKernel`, `__cudaRegisterFatBinary`, `__cudaPopCallConfiguration`).
> That code runs on your CPU to *launch* the kernel; it never touches the GPU's
> compute units. Real SASS is GPU instructions like `IMAD`, `LDG.E`, `FFMA`,
> `ISETP`, `BRA`. I explain the host stub in [§6](#6-the-host-stub-your-third-block-is-x86-not-sass),
> and give you the *actual* `sm_86` SASS in [§5](#5-sass-sm_86--the-real-machine-code).

---

## 0. The source

```cpp
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  int BLOCKSIZE = 32;
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (cRow < M && cCol < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}
```

A 1-D thread block of 1024 threads (`blockDim.x == 32*32`) computes a 32×32 tile
of `C`. The coalescing trick: consecutive `threadIdx.x` map to consecutive
`cCol` (`threadIdx.x % 32`), so a warp (32 consecutive lanes) touches 32
*consecutive* columns of `B` and of `C` → one coalesced 128-byte transaction per
access instead of 32 scattered ones.

Register terminology you'll see in PTX:

| Prefix | Type | Example |
|--------|------|---------|
| `%r`   | 32-bit int   | `%r1` |
| `%rd`  | 64-bit int (addresses) | `%rd2` |
| `%f`   | 32-bit float | `%f34` |
| `%p`   | 1-bit predicate (bool) | `%p3` |

---

## 1. Parameter loading (PTX)

```ptx
ld.param.u32   %r16, [..._param_0];   // M
ld.param.u32   %r14, [..._param_1];   // N
ld.param.u32   %r15, [..._param_2];   // K
ld.param.f32   %f8,  [..._param_3];   // alpha
ld.param.u64   %rd18,[..._param_4];   // A
ld.param.u64   %rd19,[..._param_5];   // B
ld.param.f32   %f9,  [..._param_6];   // beta
ld.param.u64   %rd17,[..._param_7];   // C
```

Kernel arguments arrive in a special **constant / parameter bank**, not
registers. `ld.param.<type>` copies them into virtual registers. Note the type
tags: `u32` for the `int`s, `f32` for the floats, `u64` for the pointers.

```ptx
cvta.to.global.u64   %rd1, %rd19;     // B: generic -> global address space
cvta.to.global.u64   %rd2, %rd18;     // A: generic -> global address space
```

`cvta.to.global` = **c**on**v**er**t** **a**ddress. A raw pointer is a "generic"
address that could point to global/shared/local; `cvta.to.global` asserts "this
is global memory" so later loads can use the cheaper `ld.global` form instead of
the generic `ld` that has to check the address space at runtime.

---

## 2. Index computation (PTX)

Source:
```cpp
cRow = blockIdx.x * 32 + (threadIdx.x / 32);
cCol = blockIdx.y * 32 + (threadIdx.x % 32);
```

PTX:
```ptx
mov.u32   %r17, %ctaid.x;      // blockIdx.x   (ctaid = CTA id = block id)
shl.b32   %r18, %r17, 5;       // blockIdx.x * 32   (<<5 == *32)
mov.u32   %r19, %tid.x;        // threadIdx.x
shr.u32   %r20, %r19, 5;       // threadIdx.x / 32  (>>5, unsigned)
add.s32   %r1,  %r18, %r20;    // cRow = blockIdx.x*32 + tid.x/32

mov.u32   %r21, %ctaid.y;      // blockIdx.y
shl.b32   %r2,  %r21, 5;       // blockIdx.y * 32
and.b32   %r3,  %r19, 31;      // threadIdx.x % 32  (& 31)
bfi.b32   %r4,  %r21, %r3, 5, 27;  // cCol, fused — see below
```

Things worth internalizing:

- `%ctaid` = **CTA id** = block index (CTA = "Cooperative Thread Array" = block).
  `%tid` = thread index, `%ntid` = blockDim, `%nctaid` = gridDim.
- **Strength reduction is already done in PTX.** `*32` became `shl ...,5` and
  `/32`, `%32` on an *unsigned* became `shr ...,5` and `and ...,31`. Because
  `threadIdx.x` is unsigned these are pure bit-ops — no expensive div/rem. (This
  is exactly why the coalesced kernel keeps `threadIdx.x` unsigned-friendly.)
- `bfi.b32 %r4, %r21, %r3, 5, 27` — **b**it **f**ield **i**nsert. It computes
  `cCol = blockIdx.y*32 + (tid.x%32)` in one instruction by inserting `blockIdx.y`
  (`%r21`) into `%r3` starting at bit position 5, width 27. Since `%r3` is already
  `tid.x % 32` (its low 5 bits), inserting `blockIdx.y` at bit 5 is arithmetically
  `%r3 + (blockIdx.y << 5)`. The compiler fused an add+shift into one op. You'll
  see the compiler pull tricks like this constantly; don't expect a 1:1 line map.

---

## 3. The bounds guard (PTX)

```cpp
if (cRow < M && cCol < N) { ... }
```

```ptx
setp.ge.s32  %p1, %r1, %r16;   // p1 = (cRow >= M)
setp.ge.s32  %p2, %r4, %r14;   // p2 = (cCol >= N)
or.pred      %p3, %p1, %p2;    // p3 = p1 || p2  = "out of bounds"
@%p3 bra     $L__BB0_9;        // if p3, jump to the ret at BB9
```

- `setp.ge.s32` = **set p**redicate, **g**reater-or-**e**qual, **s**igned **32**.
  Writes a boolean predicate register.
- De Morgan flip: `!(cRow<M && cCol<N)` becomes `(cRow>=M) || (cCol>=N)`. The
  compiler tests the *negated* condition so it can branch straight to the exit.
- `@%p3 bra` = **predicated branch**: "if `%p3` is true, `bra`nch." The `@pred`
  prefix is how PTX/SASS do conditional execution — almost every branch is guarded
  by a predicate rather than by CPU-style flags.
- `$L__BB0_9` — "**B**asic **B**lock 0, block 9". `BB0` = the 0th function.

---

## 4. The main loop (PTX) — and why it's four copies

Source is a trivial `for (i=0; i<K; ++i) tmp += A[..]*B[..];`. But `ptxas`/nvcc
**4× unrolled** it. Here's the structure:

```ptx
setp.lt.s32  %p4, %r15, 1;         // K < 1 ?
mov.f32      %f34, 0f00000000;     // tmp = 0.0f  (hex float literal)
@%p4 bra     $L__BB0_8;            // K<=0: skip loop, go straight to epilogue

add.s32      %r23, %r15, -1;       // K-1
and.b32      %r33, %r15, 3;        // K % 4   -> remainder trip count
setp.lt.u32  %p5, %r23, 3;         // (K-1) < 3, i.e. K < 4 ?
mov.u32      %r32, 0;              // i = 0
@%p5 bra     $L__BB0_5;            // K<4: skip the unrolled body, do remainder only
```

This is the classic **unroll + remainder** shape:

- `%r33 = K % 4` is how many "leftover" iterations run one-at-a-time afterward.
- If `K < 4` there's no point running the 4-wide body, so it jumps to the
  remainder loop `BB5`.

Address setup before the hot loop:
```ptx
sub.s32       %r31, %r15, %r33;    // K - (K%4) = trip count of the 4-wide loop
mul.wide.s32  %rd20, %r4, 4;       // cCol * 4 bytes   (mul.wide: 32x32 -> 64-bit)
add.s64       %rd32, %rd1, %rd20;  // &B[0*N + cCol]
mul.lo.s32    %r25, %r15, %r1;     // K * cRow
mul.wide.s32  %rd21, %r25, 4;      // (K*cRow) * 4 bytes
add.s64       %rd22, %rd2, %rd21;  // &A[cRow*K + 0]
add.s64       %rd31, %rd22, 8;     // &A[cRow*K + 2]  (base+8 bytes; see -8/-4/0/+4 below)
mul.wide.s32  %rd5,  %r14, 4;      // N * 4 bytes = stride to walk down a column of B
```

- `mul.wide.s32` multiplies two 32-bit ints producing a **64-bit** result —
  exactly what you want for `index * 4` when forming byte addresses, avoiding
  32-bit overflow.
- `%rd5 = N*4` is the byte stride to step from `B[i*N+cCol]` to `B[(i+1)*N+cCol]`
  — one row down the column.
- The `+8` on `%rd31` is a compiler trick: it bases the A-pointer in the *middle*
  of the 4-element window so the four unrolled loads become `[-8] [-4] [0] [+4]`
  offsets, which pack nicely.

The hot loop body (`BB4`), one of the four "lanes" shown:
```ptx
$L__BB0_4:
  ld.global.f32  %f14, [%rd32];        // load B[i*N + cCol]
  ld.global.f32  %f15, [%rd31+-8];     // load A[cRow*K + i]
  fma.rn.f32     %f16, %f15, %f14, %f34;  // tmp = A*B + tmp
  add.s64        %rd23, %rd32, %rd5;    // advance B pointer by N*4 (next row)
  ld.global.f32  %f17, [%rd23];        // B[(i+1)*N + cCol]
  ld.global.f32  %f18, [%rd31+-4];     // A[cRow*K + i+1]
  fma.rn.f32     %f19, %f18, %f17, %f16;
  ...                                   // two more identical (i+2, i+3)
  add.s32        %r32, %r32, 4;         // i += 4
  add.s64        %rd31, %rd31, 16;      // A ptr += 4 floats
  add.s32        %r31, %r31, -4;        // countdown -= 4
  setp.ne.s32    %p6, %r31, 0;
  @%p6 bra       $L__BB0_4;             // loop
```

Key instruction: **`fma.rn.f32 d, a, b, c` computes `d = a*b + c`** in a single
fused multiply-add. `.rn` = **r**ound to **n**earest-even (IEEE default). This is
the beating heart of GEMM — one FMA per multiply-accumulate, and it's what the
`tmp += A*B` line becomes. `%f34` is the running accumulator, threaded through the
whole chain.

Then the **remainder loop** `BB5`/`BB7` handles the last `K%4` iterations one at a
time (same body, no unroll), and falls into `BB8`.

---

### 4a. Epilogue (PTX): `alpha*tmp + beta*C`

```ptx
$L__BB0_8:
  mad.lo.s32     %r29, %r1, %r14, %r4;  // cRow*N + cCol   (mad = mul-add, integer)
  cvta.to.global.u64 %rd28, %rd17;      // C -> global
  mul.wide.s32   %rd29, %r29, 4;        // *4 bytes
  add.s64        %rd30, %rd28, %rd29;   // &C[cRow*N + cCol]
  ld.global.f32  %f27, [%rd30];         // load C
  mul.f32        %f28, %f27, %f9;       // beta * C
  fma.rn.f32     %f29, %f34, %f8, %f28; // alpha*tmp + (beta*C)
  st.global.f32  [%rd30], %f29;         // store C
$L__BB0_9:
  ret;
```

- `mad.lo.s32 d,a,b,c = (a*b + c)` truncated to low 32 bits — integer address math.
- The final result is `alpha*tmp + beta*C` done as `mul` + `fma`: `beta*C` first,
  then `fma(alpha, tmp, that)`. Same shape as the source line.

---

## 5. SASS (`sm_86`) — the real machine code

This is what `ptxas` produced for your RTX 2050. I compiled your exact kernel
with `nvcc -arch=sm_86 -cubin` and dumped it with `cuobjdump -sass`. Column at
left is the instruction address (`/*0xADDR*/`).

### 5a. Prologue — index math, compare, exit

```sass
/*0000*/  MOV R1, c[0x0][0x28] ;                 // set up stack pointer from constant bank
/*0010*/  S2R R3, SR_TID.X ;                     // R3 = threadIdx.x       (S2R = Special-reg TO Reg)
/*0020*/  S2R R5, SR_CTAID.Y ;                   // R5 = blockIdx.y
/*0030*/  S2R R4, SR_CTAID.X ;                   // R4 = blockIdx.x
/*0040*/  LOP3.LUT R0, R3, 0x1f, RZ, 0xc0, !PT ; // R0 = threadIdx.x & 31  = tid.x % 32
/*0050*/  SHF.R.U32.HI R3, RZ, 0x5, R3 ;         // R3 = threadIdx.x >> 5  = tid.x / 32
/*0060*/  SHF.L.U32 R7, R5, 0x5, RZ ;            // R7 = blockIdx.y << 5   = blockIdx.y*32
/*0070*/  LOP3.LUT R2, R7, 0xffffffe0, R0, 0xe2, !PT ; // R2 = cCol  (fused OR/AND: blockIdx.y*32 | tid%32)
/*0080*/  LEA R3, R4, R3, 0x5 ;                  // R3 = (blockIdx.x << 5) + (tid.x/32) = cRow
/*0090*/  ISETP.GE.AND P0, PT, R2, c[0x0][0x164], PT ;  // P0 = (cCol >= N)
/*00a0*/  ISETP.GE.OR  P0, PT, R3, c[0x0][0x160], P0 ;  // P0 = (cRow >= M) || P0
/*00b0*/  @P0 EXIT ;                             // out of bounds -> terminate this thread
```

SASS instructions to learn here — these recur in *every* kernel:

| SASS | Meaning |
|------|---------|
| `MOV R1, c[0x0][0x28]` | Constant bank `c[bank][offset]`. `c[0x0]` holds kernel params + ABI stuff. `0x28` is the stack base. |
| `S2R Rd, SR_TID.X` | **S**pecial-register **2** (to) **R**egister. The only way to read `threadIdx`/`blockIdx` on the GPU — they live in special registers, not the general file. |
| `LOP3.LUT` | 3-input **LO**gic **OP** driven by an 8-bit **LUT** (truth table). `0xc0` truth table = `a & b` (that's the `& 31`). `0xe2` = a fused combine of three inputs. One instruction does arbitrary boolean f(a,b,c). |
| `SHF.R.U32.HI` / `SHF.L.U32` | Funnel **SH**i**F**t right/left. Here plain `>>5` / `<<5`. |
| `LEA Rd, Ra, Rb, 0x5` | **L**oad **E**ffective **A**ddress: `Rd = (Ra << 5) + Rb`. Shift-and-add in one op (borrowed from x86 naming). Computes `cRow`. |
| `ISETP.GE.AND P0, PT, a, b, P0` | **I**nteger **SET P**redicate. `.GE` = `>=`. The trailing `.AND`/`.OR` and the last operand fold a *previous* predicate in, so `cRow>=M` and `cCol>=N` combine without a separate OR instruction. `PT` = "predicate true" (a constant-true source). |
| `@P0 EXIT` | Predicated terminate. Note it's `EXIT`, not a branch-to-ret — out-of-bounds threads just die here. |
| `RZ` | The **R**egister **Z**ero — always reads 0 (like x86 has no such thing; GPUs do). `PT`/`!PT` are the predicate equivalents (true/false). |

Notice the compiler again **fused** the index math: `bfi`/`or`/`and` from PTX
became two `LOP3.LUT` and a `LEA`. The whole `cRow`/`cCol` computation is ~6
instructions.

### 5b. Main loop — heavily unrolled, `IMAD.WIDE` + `LDG` + `FFMA`

`ptxas` unrolled the loop even more aggressively than PTX showed — there are
16-wide, 8-wide, and 4-wide chunks so it can keep many loads in flight. A slice:

```sass
/*01e0*/  IMAD.WIDE R16, R2, R17, c[0x0][0x178] ; // R16:R17 = &B[cCol]  (64-bit addr = cCol*4 + Bbase)
/*0260*/  LDG.E R35, [R16.64] ;                   // load B[...]  (LDG = LoaD Global, .E = extended/64-bit addr)
/*02a0*/  LDG.E R36, [R14.64] ;                   // load A[...]
/*02c0*/  LDG.E R31, [R14.64+0x4] ;               // load A[...+1] via immediate offset
...
/*0640*/  FFMA R33, R34, R35, R33 ;               // R33 = R34*R35 + R33   (Fused FMA)
/*0650*/  FFMA R33, R20, R37, R33 ;
/*0660*/  FFMA R31, R22, R31, R33 ;
...
/*06e0*/  @P1 BRA 0x250 ;                         // loop back while trip count high
```

The vocabulary that matters for *every* memory-bound kernel:

| SASS | Meaning |
|------|---------|
| `IMAD.WIDE Rd, Ra, Rb, Rc` | **I**nteger **M**ultiply-**ADD**, **WIDE** = 32×32→64-bit. This is the workhorse address generator: `addr = index*elemsize + base`. Replaces PTX's `mul.wide` + `add.s64`. The `.reuse` flag you see (`R9.reuse`) tells the hardware to keep that operand in the operand-reuse cache to save a register-file read — a scheduling optimization, harmless to read past. |
| `LDG.E Rd, [Raddr.64]` | **L**oa**D** from **G**lobal memory. `.E` = extended 64-bit address. `[R.64]` means the address is the 64-bit pair `R:R+1`. `+0x4`, `+0x8` are immediate byte offsets — this is how the unrolled `A[i], A[i+1]...` become one base + offsets. |
| `FFMA Rd, Ra, Rb, Rc` | **F**used **F**loat **M**ultiply-**A**dd: `Rd = Ra*Rb + Rc`. The SASS name for `fma.rn.f32`. Every `tmp += A*B` is one of these. |
| `ISETP.GT` / `@P1 BRA` | loop-condition test + predicated branch back. |
| `UIADD3 / UIADD3.X` `URx` | **U**niform integer add. `UR*` = **uniform registers**: values identical across the whole warp (e.g. a base pointer) live here so they're computed once for 32 lanes instead of per-lane. `.X` = add-with-carry (the high half of a 64-bit add). This is Ampere keeping the *scalar* part of the address math off the vector datapath. |
| `PLOP3.LUT` | Predicate version of `LOP3` — boolean algebra on predicate registers, used to manage the unroll's branch conditions. |

**What to take away for bigger kernels:** the pattern `IMAD.WIDE (make address)
→ LDG (load) → FFMA (multiply-accumulate)`, repeated and interleaved so many
loads are outstanding before the FFMAs consume them. That interleaving *is* how
the GPU hides memory latency. When you profile and see "memory bound," you're
seeing the `LDG`s stall waiting on DRAM while too few `FFMA`s are queued to hide
it — which is precisely the problem shared-memory tiling in the later kernels
fixes.

### 5c. Epilogue — `alpha*tmp + beta*C` and the store

```sass
/*0ca0*/  MOV R5, 0x4 ;
/*0cb0*/  IMAD R2, R3, c[0x0][0x164], R2 ;        // R2 = cRow*N + cCol
/*0cc0*/  IMAD.WIDE R2, R2, R5, c[0x0][0x188] ;   // &C[cRow*N+cCol]  = idx*4 + Cbase
/*0cd0*/  LDG.E R0, [R2.64] ;                      // load C
/*0ce0*/  FMUL R0, R0, c[0x0][0x180] ;             // beta * C   (beta read from constant bank)
/*0cf0*/  FFMA R37, R37, c[0x0][0x16c], R0 ;       // alpha*tmp + beta*C  (alpha from constant bank, tmp in R37)
/*0d00*/  STG.E [R2.64], R37 ;                     // store result   (STG = STore Global)
/*0d10*/  EXIT ;
/*0d20*/  BRA 0xd20 ;                              // self-branch trap (safety net after EXIT)
```

- `alpha`, `beta`, `M`, `N`, `K` are read straight from the **constant bank**
  `c[0x0][...]` — kernel scalar args never need a register until used. Offsets:
  `0x160`=M, `0x164`=N, `0x168`=K, `0x16c`=alpha, `0x180`=beta, `0x170/0x178/0x188`
  = the A/B/C pointers. (Param block starts at `0x160` on `sm_86`.)
- `FMUL` = float multiply; combined with the `FFMA` this is exactly the source's
  `alpha*tmp + beta*C`.
- `STG.E` = store to global. The final self-branch `BRA 0xd20` is a standard
  landing pad after `EXIT`.

---

## 6. The host stub (your third block is x86, not SASS)

Your third pasted block is the **CPU-side launcher** the CUDA front-end generates
so `<<<grid,block>>>(...)` becomes a real function call. It is x86-64. Two symbols
matter:

```asm
sgemm_global_mem_coalesce(...):        # the host-callable stub
    ...
    call    __cudaPopCallConfiguration  # pull grid/block/shmem/stream off the launch stack
    testl   %eax, %eax
    jne     .L3                         # if config invalid, skip
    ...
    movl    $sgemm_global_mem_coalesce, %edi   # pass the DEVICE function pointer...
    call    cudaLaunchKernel            # ...to the driver, which schedules it on the GPU
```

- `__cudaPopCallConfiguration` retrieves the `<<< >>>` config that
  `__cudaPushCallConfiguration` stashed at the call site.
- The `punpcklqdq` / `movaps ...(%rsp)` shuffling is just marshalling the 8
  arguments into the `void* args[]` array `cudaLaunchKernel` expects.
- `__sti____cudaRegisterAll` / `__cudaRegisterFunction` run at program startup
  (`atexit`/`.init`) to register the fat binary (the embedded GPU code) and bind
  the host stub symbol to the device kernel name so the driver can find the SASS.
- `__fatDeviceText` / `fatbinData` is the **fatbin**: the compiled GPU code
  (PTX and/or SASS for various arches) embedded in your executable. `1180844977`
  = `0x466243B1`, the fatbin magic number.

None of this executes on the GPU. It's the plumbing between your `host` call and
the SASS in §5.

---

## 6b. Host stub across compiler versions (old nvcc vs. CUDA 13.0)

> Still x86 host code, not GPU SASS — but comparing the two stubs is genuinely
> instructive, because the *device* SASS (§5) barely changes across `nvcc`
> versions for a kernel this simple, while the *host stub* changes visibly. Here
> the second stub is CUDA 13.0.

The GPU work is identical; what differs is how the runtime **binds the host stub
to the device kernel**. Three concrete changes:

**1. `cudaLaunchKernel` → `__cudaLaunchKernel`, via a cached handle.**

Old stub launched by passing the stub's own address directly:
```asm
movl    $sgemm_global_mem_coalesce, %edi     # device function pointer = the stub symbol
call    cudaLaunchKernel
```
CUDA 13.0 launches through a **kernel handle** it looks up once and caches:
```asm
movq    ...::__handle(%rip), %rdi            # load cached kernel handle
call    __cudaLaunchKernel                   # note the leading underscores
```
The handle is populated lazily by `__cudaGetKernel` (see below). This indirection
lets the runtime support features like per-kernel launch metadata and lazy module
loading without re-resolving the symbol on every launch.

**2. A thread-safe one-time init guard (`__cxa_guard_*`).**

The new stub wraps first-launch setup in the standard C++ static-local guard:
```asm
movzbl  guard variable ...::__tmp(%rip), %eax
testb   %al, %al
jne     .L5                    # already initialized -> skip
...
call    __cxa_guard_acquire    # take the once-init lock
...
.L14:
    movl    $sgemm_global_mem_uncoalesced, %esi
    movl    $...::__handle, %edi
    call    __cudaGetKernel    # resolve device kernel -> fill __handle, ONCE
    ...
    call    __cxa_guard_release
```
`__cxa_guard_acquire`/`_release` are the same mechanism the compiler uses for
`static` locals with runtime initializers — they guarantee `__cudaGetKernel` runs
exactly once even if two host threads launch the kernel simultaneously. The old
stub had none of this; it re-passed the symbol every call and let the runtime sort
it out.

**3. Exception-safety cold path (`[clone .cold]`, `__cxa_guard_abort`,
`_Unwind_Resume`).**

```asm
sgemm_global_mem_uncoalesced(...) [clone .cold]:
.L8:
    call    __cxa_guard_abort   # if init throws, release the guard...
    call    _Unwind_Resume      # ...and continue unwinding the exception
```
Because `__cudaGetKernel` can now fail/throw, the compiler emits a **cold section**
(`[clone .cold]` — the rarely-taken error path, placed away from the hot code for
better I-cache locality) that aborts the init guard and resumes stack unwinding.
This is why you also see `pushq %rbx` / `popq %rbx` and a slightly different frame
size (`$192` vs `$200`): `%rbx` is now a callee-saved slot holding the exception
object across the `_Unwind_Resume` call.

**Summary of the diff:**

| Aspect | Old stub | CUDA 13.0 stub |
|--------|----------|----------------|
| Launch call | `cudaLaunchKernel` (direct symbol) | `__cudaLaunchKernel` (cached `__handle`) |
| Symbol resolution | every launch | once, via `__cudaGetKernel` |
| Thread safety | none in stub | `__cxa_guard_acquire/release` |
| Error path | none | `[clone .cold]` + `__cxa_guard_abort` + `_Unwind_Resume` |
| Frame | `subq $200` | `pushq %rbx; subq $192` |

**The takeaway that generalizes:** host-stub churn between toolchain versions is
normal and cosmetic to *your* performance — it's runtime-integration plumbing.
When you compare compilers for speed, diff the **SASS** (`cuobjdump -sass`), not
this x86. For this kernel the SASS is essentially the same shape across recent
`ptxas` versions; the host stub is where the versions visibly diverge.

---

## 7. Mental model for reading harder kernels

- **Levels tell you different things.** PTX shows the compiler's *intent* and is
  target-independent; SASS shows what *actually runs* (register pressure, real
  memory instructions, occupancy limits). For performance work, read SASS.
- **Don't expect line-by-line correspondence.** Strength reduction, unrolling,
  instruction fusion (`bfi`, `LEA`, `IMAD.WIDE`, `LOP3.LUT`, `ISETP.*.OR`) mean
  one C line ↔ many/zero instructions and vice-versa.
- **The recurring GPU idioms** you now recognize and will see everywhere:
  - `S2R` to read thread/block indices.
  - `IMAD.WIDE` for 64-bit address generation.
  - `LDG`/`STG` (global), later `LDS`/`STS` (shared), `LDGSTS` (async copy on
    Ampere) — the memory instructions whose *count and pattern* determine whether
    you're coalesced and latency-bound.
  - `FFMA` — count these vs. `LDG` to reason about arithmetic intensity.
  - `@P0 BRA`/`@P0 EXIT` — predicated control flow; watch for divergence.
  - `c[0x0][...]` — constant bank for params.
  - `UR*`/`UIADD3` — uniform (per-warp scalar) datapath.
- **Coalescing, concretely:** the win in this kernel is invisible in the *count*
  of `LDG`s but visible in what addresses a warp's 32 lanes present. Because
  `cCol = tid.x % 32`, lane `t` loads `B[i*N + base + t]` — 32 consecutive floats
  → one 128-byte transaction. Swap the `/` and `%` (the uncoalesced variant) and
  the same `LDG` instruction now spans 32 rows → up to 32 transactions. Same SASS,
  32× the memory traffic. That's why you profile addresses, not instruction counts.

---

### Reproduce it yourself

```bash
nvcc -ptx   -arch=sm_86 1_coalesce.cu -o coalesce.ptx      # PTX
nvcc -cubin -arch=sm_86 1_coalesce.cu -o coalesce.cubin
cuobjdump -sass coalesce.cubin                              # real SASS
cuobjdump -ptx  coalesce.cubin                              # PTX embedded in the cubin
nvdisasm -c coalesce.cubin                                  # SASS with control-flow/CFG
```

Add `-lineinfo` to `nvcc` and `nvdisasm` will interleave source lines with SASS —
the fastest way to see which C line produced which instructions.
