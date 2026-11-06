use furiosa_opt_std::prelude::*;

// AUTO-GENERATED from rngd.dot_product — 검증된 dot_product_kernel.rs 구조 그대로 사용.
// batch_gemm 템플릿의 degenerate(M=1,N=1) 축소판이 하드웨어 제약(Time::SIZE가
// Lane::SIZE(8)로 나눠지지 않음)에 걸려 실패했기 때문에, 별도의 벡터 전체-reduction
// 전용 구조를 사용한다.
axes![A = 2048];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![1 # 256];
pub type Time = m![1];
pub type Lane = m![1];

#[device(chip = 1)]
pub fn pilot_e2e_dot_product_kernel(
    ctx: &mut Context,
    lhs: &HbmTensor<bf16, Chip, m![A]>,
    rhs: &HbmTensor<bf16, Chip, m![A]>,
) -> HbmTensor<bf16, Chip, m![1]> {
    let lhs: DmTensor<bf16, Chip, Cluster, Slice, m![A]> = lhs.to_dm(&mut ctx.tdma);
    let rhs: DmTensor<bf16, Chip, Cluster, Slice, m![A]> = rhs.to_dm(&mut ctx.tdma);

    let rhs: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![A]> = ctx
        .sub
        .begin(rhs.view())
        .fetch::<Time, m![A]>()
        .collect::<m![{ Time }, A / 16], m![A % 16]>()
        .to_trf();

    let result: DmTensor<bf16, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .main
        .begin(lhs.view())
        .fetch::<Time, m![A]>()
        .collect::<m![A / 16], m![A % 16]>()
        .contract_outer::<m![A / 32], m![A % 32], _, _, _>(&rhs)
        .contract_packet::<m![1]>()
        .contract_time::<m![1]>()
        .contract_lane::<m![1], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .commit_trim::<m![1 # 8]>()
        .commit();

    result.to_hbm(&mut ctx.tdma)
}
