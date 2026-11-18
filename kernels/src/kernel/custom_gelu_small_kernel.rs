use furiosa_opt_std::prelude::*;

// 사용자 모델 Level 3용 GELU 검증 커널 (원소 8개). pilot_e2e_gelu_erf_kernel과 같은 1패스 구성.
// gelu(x) = x * 0.5 * (1 + erf(x / √2)): x 보관 → ×1/√2 (Mul0) → erf (Fpu) → erf·0.5 + 0.5 (Fma) → ×x (Mul1)
axes![A = 8];

pub type Chip    = m![1];
pub type Cluster = m![1 # 2];
pub type Slice   = m![A / 8 # 256];

#[device(chip = 1)]
pub fn custom_gelu_small_kernel(
    ctx: &mut Device,
    input: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma);
    let result = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1, A / 4 % 2], m![A % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), std::f32::consts::FRAC_1_SQRT_2)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_ternary(FpTernaryOp::FmaF, (0.5_f32, 0.5_f32))
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![1], m![A % 8]>()
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>();
    result.to_hbm(&mut ctx.tdma)
}
