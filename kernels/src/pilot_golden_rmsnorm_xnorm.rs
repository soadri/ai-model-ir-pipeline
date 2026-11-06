use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_golden_rmsnorm_xnorm_kernel::{
    A, pilot_golden_rmsnorm_xnorm_kernel
};

mod reference_data_golden_rmsnorm_xnorm;
use reference_data_golden_rmsnorm_xnorm::{reference_x, reference_rsqrt, reference_expected};

#[tokio::main]
async fn main() { println!("golden rmsnorm xnorm test"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_pytorch_golden_reference() {
        let mut ctx = Context::acquire();

        let x_hbm = HostTensor::<f32, m![A]>::from_buf(to_bytes(&reference_x()))
            .to_hbm(&mut ctx.pdma).await;
        let r_hbm = HostTensor::<f32, m![A]>::from_buf(to_bytes(&reference_rsqrt()))
            .to_hbm(&mut ctx.pdma).await;

        let out_hbm = launch(
            pilot_golden_rmsnorm_xnorm_kernel, (&mut ctx, &x_hbm, &r_hbm)
        ).await;
        let actual: Vec<f32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.into_vec();
        let expected = reference_expected();

        println!("=== RMSNorm x_norm — Golden Reference 검증 ===");
        for i in 0..4 {
            println!("  [{i}] PyTorch={:.4}  RNGD={:.4}  diff={:.6}",
                expected[i], actual[i], (expected[i] - actual[i]).abs());
        }

        let mut fail = 0;
        for i in 0..A::SIZE {
            let tol = (expected[i].abs() * 0.02_f32).max(1e-4_f32);
            if (expected[i] - actual[i]).abs() > tol { fail += 1; }
        }
        assert!(fail == 0, "Golden Reference 불일치: {fail}/{}", A::SIZE);
        println!("✅ PASS: RMSNorm x_norm Golden Reference 일치");
    }
}
