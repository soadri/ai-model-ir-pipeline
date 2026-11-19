use furiosa_opt_std::prelude::*;

// 사용자 모델 Level 3용 ReLU 검증 커널 (원소 8개). pilot_e2e_relu_kernel과 같은 구성: Clip ALU의 max(x, 0) 하나.

axes![A = 8];

pub type Chip    = m![1];
pub type Cluster = m![1 # 2];
pub type Slice   = m![A / 8 # 256];

#[device(chip = 1)]
pub fn custom_relu_small_kernel(
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
        .vector_clip(ClipBinaryOpF32::Max, 0.0_f32)
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>();
    result.to_hbm(&mut ctx.tdma)
}
