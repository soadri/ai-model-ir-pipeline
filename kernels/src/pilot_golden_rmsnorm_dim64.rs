use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_golden_rmsnorm_mul_kernel::{
    A, pilot_golden_rmsnorm_mul_kernel
};
use rngd_tcp_kernel_dev::kernel::pilot_golden_rmsnorm_rsqrt_kernel::{
    pilot_golden_rmsnorm_rsqrt_kernel
};

mod reference_data_rmsnorm_dim64;
use reference_data_rmsnorm_dim64::*;

#[tokio::main]
async fn main() { println!("RMSNorm dim=64 tiling test (0.6)"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rmsnorm_dim64_tiling() {
        let mut ctx = Context::acquire();
        let input = reference_input();
        let expected = reference_expected();
        let eps: f32 = 1e-05;

        // Step 1: x² (NPU, 4청크)
        let mut x_sq_all = Vec::new();
        for t in 0..4 {
            let start = t * 16;
            let chunk: Vec<f32> = input[start..start+16].to_vec();
            let x_hbm = HostTensor::<f32, m![A]>::from_buf(to_bytes(&chunk))
                .to_hbm(&mut ctx.pdma).await;
            let xsq_hbm = launch(pilot_golden_rmsnorm_mul_kernel, (&mut ctx, &x_hbm)).await;
            let xsq: Vec<f32> = xsq_hbm.to_host::<m![A]>(&mut ctx.pdma).await.into_vec();
            x_sq_all.extend(xsq);
        }
        println!("Step 1 x_sq[0:4]: {:?}", &x_sq_all[..4]);

        // Step 2: mean_sq + rsqrt (CPU)
        let mean_sq = x_sq_all.iter().sum::<f32>() / 64_f32;
        let rsqrt_val = 1.0_f32 / (mean_sq + eps).sqrt();
        println!("Step 2 mean_sq={:.6} rsqrt={:.6}", mean_sq, rsqrt_val);

        // Step 3: x * rsqrt (CPU)
        let mut result = Vec::new();
        for t in 0..4 {
            let start = t * 16;
            let normed: Vec<f32> = input[start..start+16].iter()
                .map(|&v| v * rsqrt_val).collect();
            result.extend(normed);
        }

        println!("\n=== RMSNorm dim=64 (0.6) ===");
        let mut max_diff: f32 = 0.0;
        for i in 0..4 {
            let diff = (expected[i] - result[i]).abs();
            if diff > max_diff { max_diff = diff; }
            println!("  [{i}] expected={:.4}  actual={:.4}  diff={:.6}", expected[i], result[i], diff);
        }
        for i in 0..64 {
            let diff = (expected[i] - result[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }
        assert!(max_diff < 0.001, "mismatch: max_diff={max_diff}");
        println!("PASS: max_diff={:.6}", max_diff);
    }
}
