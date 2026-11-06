use furiosa_opt_std::prelude::*;

// AUTO-GENERATED from rngd.elementwise(op="pow2") — 단항(unary), EXPERIMENTAL
// 하드웨어엔 Pow가 없어서 같은 입력을 두 번 적재해 MulF(x,x)로 조합.
axes![A = 2048];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![A / 8 # 256];

#[device(chip = 1)]
pub fn pilot_e2e_pow2_kernel(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let input_dm = input.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma);

    // 같은 입력을 VRF로도 적재 — 자기 자신과 곱하기 위함
    let self_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![A % 8]> = ctx
        .sub
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![A % 8 / 8], m![A % 8 % 8]>()
        .to_vrf();

    let result = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1, A / 4 % 2], m![A % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &self_vrf)
        .vector_widen_concat::<m![1], m![A % 8]>()
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>();

    result.to_hbm(&mut ctx.tdma)
}
