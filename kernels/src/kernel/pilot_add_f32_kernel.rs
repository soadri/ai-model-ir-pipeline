use furiosa_opt_std::prelude::*;

axes![A = 2048];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![A / 8 # 256];

// elementwise_mul_kernel.rs 구조를 그대로 사용, 다음 두 곳만 교체:
//   1. dtype: i32 -> f32
//   2. 연산자: FxpBinaryOp::MulInt -> FpBinaryOp::AddF
// vector_fp_binary가 vector_fxp와 동일한 인자 규칙(스칼라 또는 VRF 텐서)을
// 따른다는 문서 근거(Binary/ternary 카테고리 공통 서술)에 기반한 추정.
// confirmed=False — 아래 컴파일/테스트로 실증 필요.
#[device(chip = 1)]
pub fn pilot_add_f32_kernel(
    ctx: &mut Context,
    lhs: &HbmTensor<f32, Chip, m![A]>,
    rhs: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {
    let lhs_dm = lhs.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma);
    let rhs_dm = rhs.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma);

    let rhs_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![A % 8]> = ctx
        .sub
        .begin(rhs_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![A % 8 / 8], m![A % 8 % 8]>()
        .to_vrf();

    let result = ctx
        .main
        .begin(lhs_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        // narrow_trim이 아니라 narrow_split: 뒤 4레인도 실데이터라서
        // A%8 = (A/4%2)*4 + (A%4) 로 쪼개, 앞의 절반 선택 비트를 Time에 흡수
        .vector_narrow_split::<m![1, A / 4 % 2], m![A % 4]>()
        // 여기가 핵심 검증 대상: f32 텐서-텐서 add가 실제로 되는지
        .vector_fp_binary(FpBinaryOp::AddF, &rhs_vrf)
        // narrow_split의 역연산은 widen_concat (widen_pad 아님)
        .vector_widen_concat::<m![1], m![A % 8]>()
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>();

    result.to_hbm(&mut ctx.tdma)
}
