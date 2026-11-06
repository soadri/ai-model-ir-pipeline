use furiosa_opt_std::prelude::*;

// Golden Reference — RMSNorm x_norm = x * rsqrt_val
// 입력1: x [A], 입력2: rsqrt_val [A] (브로드캐스트 스칼라)
axes![A = 16];

pub type Chip    = m![1];
pub type Cluster = m![1 # 2];
pub type Slice   = m![A / 8 # 256];

#[device(chip = 1)]
pub fn pilot_golden_rmsnorm_xnorm_kernel(
    ctx: &mut Context,
    x:         &HbmTensor<f32, Chip, m![A]>,
    rsqrt_val: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let x_dm = x.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma);
    let r_dm  = rsqrt_val.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma);

    let r_vrf = ctx
        .sub
        .begin(r_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .to_vrf::<m![A % 8]>();

    let result = ctx
        .main
        .begin(x_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1, A / 4 % 2], m![A % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &r_vrf)
        .vector_widen_concat::<m![1], m![A % 8]>()
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>();

    result.to_hbm(&mut ctx.tdma)
}
