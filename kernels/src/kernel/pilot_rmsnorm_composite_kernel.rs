use furiosa_opt_std::prelude::*;

// FFN composite: silu(a) * b — elementwise 복합 연산 (0.6 실험)
// SiLU = x * sigmoid(x)
// 최종: silu(a) * b = a * sigmoid(a) * b
axes![A = 2048];

pub type Chip    = m![1];
pub type Cluster = m![1 # 2];
pub type Slice   = m![A / 8 # 256];

// Step 1: silu(a) = a * sigmoid(a)
#[device(chip = 1)]
pub fn silu_kernel(
    ctx: &mut Context,
    a: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let a_dm = a.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma);

    let result = ctx
        .main
        .begin(a_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1, A / 4 % 2], m![A % 4]>()
        .vector_fp_unary(FpUnaryOp::Sigmoid)
        .vector_widen_concat::<m![1], m![A % 8]>()
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>();

    // silu = a * sigmoid(a) → 2패스 필요
    // Pass 2: a * sigmoid_result
    let sig_vrf = ctx
        .sub
        .begin(result.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .to_vrf::<m![A % 8]>();

    let silu: DmTensor<f32, Chip, Cluster, Slice, m![A % 8]> = ctx
        .main
        .begin(a_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1, A / 4 % 2], m![A % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sig_vrf)
        .vector_widen_concat::<m![1], m![A % 8]>()
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>();

    silu.to_hbm(&mut ctx.tdma)
}

// Step 2: silu_result * gate (elementwise mul)
#[device(chip = 1)]
pub fn gate_mul_kernel(
    ctx: &mut Context,
    silu_out: &HbmTensor<f32, Chip, m![A]>,
    gate:     &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let s_dm = silu_out.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma);
    let g_dm = gate.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma);

    let g_vrf = ctx
        .sub
        .begin(g_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .to_vrf::<m![A % 8]>();

    let result: DmTensor<f32, Chip, Cluster, Slice, m![A % 8]> = ctx
        .main
        .begin(s_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1, A / 4 % 2], m![A % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &g_vrf)
        .vector_widen_concat::<m![1], m![A % 8]>()
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>();

    result.to_hbm(&mut ctx.tdma)
}
