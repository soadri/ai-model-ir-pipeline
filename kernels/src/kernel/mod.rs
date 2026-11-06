pub mod constant_add_kernel;
pub mod dot_product_kernel;
pub mod elementwise_mul_kernel;
pub mod gemm_kernel;
pub mod gemv_kernel;
pub mod pilot_add_f32_kernel;
pub mod pilot_e2e_add_kernel;
pub mod pilot_e2e_sub_kernel;
pub mod pilot_e2e_mul_kernel;
pub mod pilot_batch_gemm_kernel;  // TODO: 재작성 필요  // TODO: Phase 3 — flit/slice 크기 재설계 필요
pub mod pilot_e2e_batch_gemm_kernel;  // TODO: 재작성 필요  // TODO: Cluster config 재설계 필요
pub mod pilot_e2e_gemm_kernel;  // TODO: Cluster config 재설계 필요
pub mod pilot_e2e_div_kernel;
pub mod pilot_e2e_dot_product_kernel;
pub mod pilot_e2e_rsqrt_kernel;
pub mod pilot_e2e_pow2_kernel;
pub mod pilot_e2e_sqrt_kernel;
pub mod pilot_e2e_exp_kernel;
pub mod pilot_e2e_sigmoid_kernel;
pub mod pilot_e2e_log_kernel;
pub mod pilot_e2e_tanh_kernel;
pub mod pilot_e2e_sin_kernel;
pub mod pilot_e2e_cos_kernel;
pub mod pilot_e2e_gemv_kernel;
pub mod pilot_e2e_erf_kernel;
// // pub mod pilot_e2e_transpose_kernel;  // TODO: vISA lowering 재설계  // TODO: vISA lowering 에러
// // pub mod pilot_e2e_reduce_mean_kernel;  // TODO: Phase 3 — flit/slice 크기 재설계 필요
pub mod pilot_e2e_silu_kernel;
// // pub mod pilot_e2e_gelu_kernel;  // TODO: Stash API 변경 대응 필요
pub mod pilot_e2e_gemm_large_kernel;  // TODO: 재작성 필요  // TODO: Phase 3 — flit/slice 크기 재설계 필요
pub mod pilot_golden_ffn_gate_kernel;  // TODO: 재작성 필요  // TODO: Cluster config 재설계 필요
pub mod pilot_golden_ffn_down_kernel;  // TODO: 재작성 필요  // TODO: Cluster config 재설계 필요
pub mod pilot_golden_silu_small_kernel;
pub mod pilot_golden_rmsnorm_mul_kernel;
pub mod pilot_golden_rmsnorm_div_kernel;
pub mod pilot_golden_rmsnorm_rsqrt_kernel;
pub mod pilot_golden_rmsnorm_xnorm_kernel;
pub mod pilot_golden_attn_wo_kernel;  // TODO: 재작성 필요  // TODO: Cluster config 재설계 필요
pub mod pilot_golden_attn_qkt_kernel;  // TODO: 재작성 필요  // TODO: Cluster config 재설계 필요
pub mod pilot_golden_attn_sv_kernel;  // TODO: 재작성 필요  // TODO: Cluster config 재설계 필요
pub mod pilot_golden_ffn_down_k8_kernel;  // TODO: 재작성 필요  // TODO: Phase 3 — flit/slice 크기 재설계 필요
pub mod pilot_golden_ffn_down_n16_kernel;  // TODO: 재작성 필요  // TODO: Phase 3 — flit/slice 크기 재설계 필요
pub mod pilot_golden_add_a16_kernel;
pub mod pilot_golden_add_a8_kernel;
// // // pub mod pilot_golden_intra_reduce_kernel;  // TODO: Phase 3 — flit/slice 크기 재설계 필요

pub mod pilot_rmsnorm_composite_kernel;
pub mod test_k128_gemm_kernel;
pub mod test_k256_gemm_kernel;
pub mod gemm_m16_k256_kernel;
