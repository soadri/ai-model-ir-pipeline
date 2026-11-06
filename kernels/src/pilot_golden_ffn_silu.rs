use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_golden_silu_small_kernel::{
    A, pilot_golden_silu_small_kernel
};

mod reference_data_golden_ffn_silu;
use reference_data_golden_ffn_silu::{reference_input, reference_expected};

#[tokio::main]
async fn main() { println!("golden ffn silu test (0.6)"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_pytorch_golden_reference() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<f32, m![A]>::from_buf(to_bytes(&reference_input()));
        let input_hbm = input.to_hbm(&mut ctx.pdma).await;

        let out_hbm = launch(pilot_golden_silu_small_kernel, (&mut ctx, &input_hbm)).await;
        let actual: Vec<f32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.into_vec();
        let expected = reference_expected();

        println!("=== FFN SiLU — Golden Reference (0.6) ===");
        for i in 0..A::SIZE {
            println!("  [{i}] PyTorch={:.4}  RNGD={:.4}  diff={:.6}",
                expected[i], actual[i], (expected[i] - actual[i]).abs());
        }

        let mut fail = 0;
        for i in 0..A::SIZE {
            let tol = (expected[i].abs() * 0.02_f32).max(1e-4_f32);
            if (expected[i] - actual[i]).abs() > tol { fail += 1; }
        }
        assert!(fail == 0, "Golden Reference mismatch: {fail}/{}", A::SIZE);
        println!("PASS: SiLU Golden Reference");
    }
}
