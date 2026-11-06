use furiosa_opt_std::prelude::*;

// FFN down linear — gemm_kernel 패턴 기반 재작성 (0.6)
axes![M = 512, K = 64, N = 512];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![M / 32, N / 32];
pub type Lane = m![N % 8];

#[device(chip = 1)]
pub fn pilot_golden_ffn_down_kernel(
    ctx: &mut Context,
    a: &HbmTensor<bf16, Chip, m![M, K]>,
    b: &HbmTensor<bf16, Chip, m![N, K]>,
) -> HbmTensor<bf16, Chip, m![M, N]> {
    let a: DmTensor<bf16, Chip, Cluster, Slice, m![M % 32, K]> = a.to_dm(&mut ctx.tdma);
    let b: DmTensor<bf16, Chip, Cluster, Slice, m![N % 32, K]> = b.to_dm(&mut ctx.tdma);

    let b_trf: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![N / 8 % 4, K]> = ctx
        .sub
        .begin(b.view())
        .fetch::<m![N % 8, N / 8 % 4], m![K]>()
        .collect::<m![N % 8, N / 8 % 4, K / 16], m![K % 16]>()
        .to_trf();

    let result: DmTensor<bf16, Chip, Cluster, Slice, m![M % 32, N % 32]> = ctx
        .main
        .begin(a.view())
        .fetch::<m![M % 32, N / 8 % 4], m![K]>()
        .collect::<m![M % 32, N / 8 % 4, K / 16], m![K % 16]>()
        .contract_outer::<m![M % 32, N / 8 % 4, K / 32], m![K % 32], _, _, _>(&b_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![M % 32, N / 8 % 4]>()
        .contract_lane::<m![M % 32, N / 8 % 4], m![N % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![N % 8 # 16]>()
        .commit_trim::<m![N % 8]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}
