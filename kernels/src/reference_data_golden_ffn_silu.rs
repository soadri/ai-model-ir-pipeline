use furiosa_opt_std::prelude::*;

// Golden Reference — FFN SiLU
// 입력: gate_out [1,8], 출력: silu_out [1,8]
pub const A: usize = 8;

const INPUT_F32:  &[f32] = &[-0.71093750_f32, -4.37500000_f32, -2.29687500_f32, 1.51562500_f32, 2.87500000_f32, -0.95703125_f32, -4.15625000_f32, 1.14843750_f32];
const EXPECT_F32: &[f32] = &[-0.23437500_f32, -0.05444336_f32, -0.20996094_f32, 1.24218750_f32, 2.71875000_f32, -0.26562500_f32, -0.06396484_f32, 0.87109375_f32];

pub fn reference_input()    -> Vec<f32> { INPUT_F32.to_vec() }
pub fn reference_expected() -> Vec<f32> { EXPECT_F32.to_vec() }
