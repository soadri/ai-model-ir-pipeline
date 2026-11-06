use furiosa_opt_std::prelude::*;

// mean(x²) — intra_slice_reduce 패턴 (0.6)
axes![A = 2048];

pub type Chip    = m![1];
pub type Cluster = m![1 # 2];
pub type Slice   = m![1 # 256];

// Pass 1: sum(x²)
#[device(chip = 1)]
pub fn reduce_sum_sq_kernel(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![1]> {
    let input_dm: DmTensor<f32, Chip, Cluster, Slice, m![A]> = input.to_dm(&mut ctx.tdma);

    let input_vrf = ctx
        .sub
        .begin(input_dm.view())
        .fetch::<m![1], m![A]>()
        .collect::<m![A / 8], m![A % 8]>()
        .to_vrf::<m![A % 8]>();

    let result = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![A]>()
        .collect::<m![A / 8], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1, A / 4 % 2], m![A % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_vrf)
        .vector_intra_slice_reduce::<A, m![1], m![1]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![2]>()
        .vector_final()
        .commit_trim::<m![2]>()
        .commit::<m![2]>();

    result.to_hbm(&mut ctx.tdma)
}

// Pass 2: scale by inv_n
#[device(chip = 1)]
pub fn scale_kernel(
    ctx: &mut Context,
    sum_sq: &HbmTensor<f32, Chip, m![A]>,
    inv_n:  &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let sum_dm = sum_sq.to_dm::<Cluster, m![A / 8 # 256], m![A % 8]>(&mut ctx.tdma);
    let inv_dm = inv_n.to_dm::<Cluster, m![A / 8 # 256], m![A % 8]>(&mut ctx.tdma);

    let inv_vrf = ctx
        .sub
        .begin(inv_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .to_vrf::<m![A % 8]>();

    let result = ctx
        .main
        .begin(sum_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1, A / 4 % 2], m![A % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inv_vrf)
        .vector_widen_concat::<m![1], m![A % 8]>()
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>();

    result.to_hbm(&mut ctx.tdma)
}
