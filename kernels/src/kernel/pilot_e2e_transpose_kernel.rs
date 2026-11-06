use furiosa_opt_std::prelude::*;

// Transpose bf16 — 0.6 lowering 제약 기반
// bf16: time=2, packet=16(32B)
axes![M = 32, N = 256];

pub type Chip    = m![1];
pub type Cluster = m![1 # 2];
pub type Slice   = m![M / 2, N / 16];  // 16 * 16 = 256 ✅

#[device(chip = 1)]
pub fn pilot_e2e_transpose_kernel(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![M, N]>,
) -> HbmTensor<bf16, Chip, m![N, M]> {
    let input_dm = input.to_dm::<Cluster, Slice, m![M % 2, N % 16]>(&mut ctx.tdma);

    let result: DmTensor<bf16, Chip, Cluster, Slice, m![N % 16, M % 2]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![M % 2], m![N % 16]>()
        .collect::<m![M % 2], m![N % 16]>()
        .transpose::<m![8], m![4]>()
        .commit_trim::<m![4]>()
        .commit::<m![N % 16, M % 2]>();

    result.to_hbm(&mut ctx.tdma)
}
