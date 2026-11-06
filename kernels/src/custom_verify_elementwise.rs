use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_golden_rmsnorm_mul_kernel::{
    A, pilot_golden_rmsnorm_mul_kernel
};
use rngd_tcp_kernel_dev::kernel::pilot_golden_rmsnorm_rsqrt_kernel::{
    pilot_golden_rmsnorm_rsqrt_kernel
};
use rngd_tcp_kernel_dev::kernel::pilot_golden_rmsnorm_xnorm_kernel::{
    A as AX, pilot_golden_rmsnorm_xnorm_kernel
};

fn load_f32_bin(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect(&format!("failed to read {}", path));
    bytes.chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[tokio::main]
async fn main() { println!("custom elementwise verify"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn custom_pow2() {
        let mut ctx = Context::acquire();
        let input = load_f32_bin("data/custom_elem_input.bin");
        let expected = load_f32_bin("data/custom_elem_pow2_expected.bin");

        let x_hbm = HostTensor::<f32, m![A]>::from_buf(to_bytes(&input))
            .to_hbm(&mut ctx.pdma).await;
        let out_hbm = launch(pilot_golden_rmsnorm_mul_kernel, (&mut ctx, &x_hbm)).await;
        let result: Vec<f32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.into_vec();

        println!("=== custom pow2 ===");
        let mut max_diff: f32 = 0.0;
        for i in 0..A::SIZE {
            let diff = (expected[i] - result[i]).abs();
            if diff > max_diff { max_diff = diff; }
            if i < 4 {
                println!("  [{}] expected={:.6} actual={:.6} diff={:.6}", i, expected[i], result[i], diff);
            }
        }
        assert!(max_diff < 0.001, "pow2 mismatch: max_diff={max_diff}");
        println!("PASS: pow2 max_diff={:.6}", max_diff);
    }

    #[tokio::test]
    async fn custom_rsqrt() {
        let mut ctx = Context::acquire();
        let input = load_f32_bin("data/custom_elem_rsqrt_input.bin");
        let expected = load_f32_bin("data/custom_elem_rsqrt_expected.bin");

        let x_hbm = HostTensor::<f32, m![A]>::from_buf(to_bytes(&input))
            .to_hbm(&mut ctx.pdma).await;
        let out_hbm = launch(pilot_golden_rmsnorm_rsqrt_kernel, (&mut ctx, &x_hbm)).await;
        let result: Vec<f32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.into_vec();

        println!("=== custom rsqrt ===");
        let mut max_diff: f32 = 0.0;
        for i in 0..A::SIZE {
            let diff = (expected[i] - result[i]).abs();
            if diff > max_diff { max_diff = diff; }
            if i < 4 {
                println!("  [{}] expected={:.6} actual={:.6} diff={:.6}", i, expected[i], result[i], diff);
            }
        }
        assert!(max_diff < 0.001, "rsqrt mismatch: max_diff={max_diff}");
        println!("PASS: rsqrt max_diff={:.6}", max_diff);
    }

    #[tokio::test]
    async fn custom_xnorm() {
        let mut ctx = Context::acquire();
        let x = load_f32_bin("data/custom_elem_xnorm_x.bin");
        let rsqrt = load_f32_bin("data/custom_elem_xnorm_rsqrt.bin");
        let expected = load_f32_bin("data/custom_elem_xnorm_expected.bin");

        let x_hbm = HostTensor::<f32, m![AX]>::from_buf(to_bytes(&x))
            .to_hbm(&mut ctx.pdma).await;
        let r_hbm = HostTensor::<f32, m![AX]>::from_buf(to_bytes(&rsqrt))
            .to_hbm(&mut ctx.pdma).await;
        let out_hbm = launch(pilot_golden_rmsnorm_xnorm_kernel, (&mut ctx, &x_hbm, &r_hbm)).await;
        let result: Vec<f32> = out_hbm.to_host::<m![AX]>(&mut ctx.pdma).await.into_vec();

        println!("=== custom xnorm ===");
        let mut max_diff: f32 = 0.0;
        for i in 0..AX::SIZE {
            let diff = (expected[i] - result[i]).abs();
            if diff > max_diff { max_diff = diff; }
            if i < 4 {
                println!("  [{}] expected={:.6} actual={:.6} diff={:.6}", i, expected[i], result[i], diff);
            }
        }
        assert!(max_diff < 0.001, "xnorm mismatch: max_diff={max_diff}");
        println!("PASS: xnorm max_diff={:.6}", max_diff);
    }
}
