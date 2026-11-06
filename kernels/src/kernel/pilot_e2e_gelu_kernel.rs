use furiosa_opt_std::prelude::*;

// GELU 3패스 구현
// Pass 1: x → x² → x³ → x³*0.044715
// Pass 2: (x + x³*0.044715) * √(2/π) → tanh
// Pass 3: (1 + tanh) * 0.5 * x
axes![A = 2048];

pub type Chip    = m![1];
pub type Cluster = m![1 # 2];
pub type Slice   = m![A / 8 # 256];

// Pass 1: x → x³ * 0.044715
#[device(chip = 1)]
pub fn gelu_pass1_kernel(
    ctx: &mut Context,
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
        // x² (Mul0)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        // x³ (Mul1)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        // x³ * 0.044715 (Fma)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Fma), 0.044715_f32)
        .vector_widen_concat::<m![1], m![A % 8]>()
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>();

    result.to_hbm(&mut ctx.tdma)
}

// Pass 2: (x + x³*0.044715) * √(2/π) → tanh
#[device(chip = 1)]
pub fn gelu_pass2_kernel(
    ctx: &mut Context,
    input:     &HbmTensor<f32, Chip, m![A]>,  // 원본 x
    cubic_term: &HbmTensor<f32, Chip, m![A]>, // x³ * 0.044715
) -> HbmTensor<f32, Chip, m![A]> {
    let input_dm  = input.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma);
    let cubic_dm  = cubic_term.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma);

    let cubic_vrf = ctx
        .sub
        .begin(cubic_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .to_vrf::<m![A % 8]>();

    let result = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1, A / 4 % 2], m![A % 4]>()
        // x + x³*0.044715 (VRF)
        .vector_fp_binary(FpBinaryOp::AddF, &cubic_vrf)
        // * √(2/π) (Mul0)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.7978845608_f32)
        // tanh
        .vector_fp_unary(FpUnaryOp::Tanh)
        .vector_widen_concat::<m![1], m![A % 8]>()
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>();

    result.to_hbm(&mut ctx.tdma)
}

// Pass 3: (1 + tanh) * 0.5 * x
#[device(chip = 1)]
pub fn gelu_pass3_kernel(
    ctx: &mut Context,
    tanh_val: &HbmTensor<f32, Chip, m![A]>,  // tanh 결과
    input:    &HbmTensor<f32, Chip, m![A]>,  // 원본 x
) -> HbmTensor<f32, Chip, m![A]> {
    let tanh_dm  = tanh_val.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma);
    let input_dm = input.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma);

    let input_vrf = ctx
        .sub
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .to_vrf::<m![A % 8]>();

    let result = ctx
        .main
        .begin(tanh_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1, A / 4 % 2], m![A % 4]>()
        // 1 + tanh
        .vector_fp_binary(FpBinaryOp::AddF, 1.0_f32)
        // * 0.5 (Mul0)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.5_f32)
        // * x (VRF, Mul1)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &input_vrf)
        .vector_widen_concat::<m![1], m![A % 8]>()
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>();

    result.to_hbm(&mut ctx.tdma)
}
