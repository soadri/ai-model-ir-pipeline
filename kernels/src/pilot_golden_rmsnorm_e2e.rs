use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_golden_rmsnorm_mul_kernel::{
    A, pilot_golden_rmsnorm_mul_kernel
};
use rngd_tcp_kernel_dev::kernel::pilot_golden_rmsnorm_rsqrt_kernel::{
    pilot_golden_rmsnorm_rsqrt_kernel
};
use rngd_tcp_kernel_dev::kernel::pilot_golden_rmsnorm_xnorm_kernel::{
    pilot_golden_rmsnorm_xnorm_kernel
};

mod reference_data_golden_rmsnorm_e2e;
use reference_data_golden_rmsnorm_e2e::{reference_input, reference_rsqrt, reference_expected};

#[tokio::main]
async fn main() { println!("RMSNorm end-to-end test (0.6)"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rmsnorm_end_to_end() {
        let mut ctx = Context::acquire();

        let x = reference_input();
        let expected = reference_expected();

        let x_hbm = HostTensor::<f32, m![A]>::from_buf(to_bytes(&x))
            .to_hbm(&mut ctx.pdma).await;

        // Step 1: x_sq = x * x (RNGD)
        let xsq_hbm = launch(pilot_golden_rmsnorm_mul_kernel, (&mut ctx, &x_hbm)).await;
        let x_sq: Vec<f32> = xsq_hbm
            .to_host::<m![A]>(&mut ctx.pdma).await.into_vec();
        println!("Step 1 x_sq[0:4]: {:?}", &x_sq[..4]);

        // Step 2: mean_sq = sum(x_sq) / N (CPU fallback)
        let mean_sq = x_sq.iter().sum::<f32>() / A::SIZE as f32;
        let eps = 1e-6_f32;
        let rsqrt_val = 1.0_f32 / (mean_sq + eps).sqrt();
        println!("Step 2 mean_sq(CPU):{:.6}  rsqrt:{:.6}", mean_sq, rsqrt_val);

        // Step 3: rsqrt (RNGD)
        let rms_vec: Vec<f32> = vec![(mean_sq + eps); A::SIZE];
        let rms_hbm = HostTensor::<f32, m![A]>::from_buf(to_bytes(&rms_vec))
            .to_hbm(&mut ctx.pdma).await;
        let rsqrt_hbm = launch(pilot_golden_rmsnorm_rsqrt_kernel, (&mut ctx, &rms_hbm)).await;
        let rsqrt_rngd: Vec<f32> = rsqrt_hbm
            .to_host::<m![A]>(&mut ctx.pdma).await.into_vec();
        println!("Step 3 rsqrt RNGD:{:.6}  CPU:{:.6}", rsqrt_rngd[0], rsqrt_val);

        // Step 4: x_norm = x * rsqrt_val (RNGD)
        use rngd_tcp_kernel_dev::kernel::pilot_golden_rmsnorm_xnorm_kernel::A as AX;
        let x_hbm2 = HostTensor::<f32, m![AX]>::from_buf(to_bytes(&x))
            .to_hbm(&mut ctx.pdma).await;
        let rsqrt_hbm2 = HostTensor::<f32, m![AX]>::from_buf(to_bytes(&rsqrt_rngd))
            .to_hbm(&mut ctx.pdma).await;
        let xnorm_hbm = launch(
            pilot_golden_rmsnorm_xnorm_kernel, (&mut ctx, &x_hbm2, &rsqrt_hbm2)
        ).await;
        let x_norm: Vec<f32> = xnorm_hbm
            .to_host::<m![AX]>(&mut ctx.pdma).await.into_vec();
        println!("Step 4 x_norm[0:4]: {:?}", &x_norm[..4]);

        // 검증
        println!("\n=== RMSNorm End-to-End (0.6) ===");
        for i in 0..A::SIZE {
            println!("  [{i:2}] PyTorch={:8.4}  RNGD={:8.4}  diff={:.6}",
                expected[i], x_norm[i],
                (expected[i] - x_norm[i]).abs());
        }

        let mut fail = 0;
        for i in 0..A::SIZE {
            let tol = (expected[i].abs() * 0.02_f32).max(1e-4_f32);
            if (expected[i] - x_norm[i]).abs() > tol { fail += 1; }
        }
        assert!(fail == 0, "RMSNorm e2e mismatch: {fail}/{}", A::SIZE);
        println!("PASS: RMSNorm End-to-End");
    }
}
