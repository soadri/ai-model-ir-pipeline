use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_golden_rmsnorm_rsqrt_kernel::{A, pilot_golden_rmsnorm_rsqrt_kernel};

mod reference_data_golden_rmsnorm_rsqrt;
use reference_data_golden_rmsnorm_rsqrt::{reference_input, reference_expected};

#[tokio::main]
async fn main() { println!("golden rmsnorm rsqrt test"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_pytorch_golden_reference() {
        let mut ctx = Context::acquire();
        let input = HostTensor::<f32, m![A]>::from_buf(to_bytes(&reference_input()));
        let input_hbm = input.to_hbm(&mut ctx.pdma).await;

        let out_hbm = launch(pilot_golden_rmsnorm_rsqrt_kernel, (&mut ctx, &input_hbm)).await;
        let actual: Vec<f32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.into_vec();
        let expected = reference_expected();

        println!("=== RMSNorm rsqrt(rms) — Golden Reference 검증 ===");
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
        println!("✅ PASS: RMSNorm rsqrt(rms) Golden Reference 일치");
    }
}
