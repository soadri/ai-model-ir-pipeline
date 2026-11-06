use furiosa_opt_std::prelude::*;

// f32 reduce — vector_intra_slice_reduce 패턴 (0.6)
// 데이터를 1 slice에 넣고 slice 내부에서 reduce
axes![A = 2048];

pub type Chip    = m![1];
pub type Cluster = m![1 # 2];
pub type Slice   = m![1 # 256];

#[device(chip = 1)]
pub fn pilot_golden_intra_reduce_kernel(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![1]> {
    let input_dm: DmTensor<f32, Chip, Cluster, Slice, m![A]> = input.to_dm(&mut ctx.tdma);

    let result = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![A]>()
        .collect::<m![A / 8], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_reduce::<A, m![1], m![1]>(IntraSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1, 1], m![1]>()
        .vector_widen_pad::<m![1]>()
        .vector_final()
        .commit_trim::<m![1 # 4]>()
        .commit::<m![1 # 4]>();

    result.to_hbm(&mut ctx.tdma)
}
