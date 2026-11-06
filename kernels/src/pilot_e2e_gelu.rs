use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_gelu_kernel::{
    A, gelu_pass1_kernel, gelu_pass2_kernel, gelu_pass3_kernel
};

mod reference_data_e2e_gelu;
use reference_data_e2e_gelu::{CHECK_N, reference_input, reference_expected};

#[tokio::main]
async fn main() { println!("gelu kernel test"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_actual_pytorch_output() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<f32, m![A]>::from_buf(to_bytes(&reference_input()));
        let input_hbm = input.to_hbm(&mut ctx.pdma).await;

        // Pass 1: x³ * 0.044715
        let cubic_hbm = launch(gelu_pass1_kernel, (&mut ctx, &input_hbm)).await;

        // Pass 2: (x + x³*0.044715) * √(2/π) → tanh
        let tanh_hbm = launch(gelu_pass2_kernel, (&mut ctx, &input_hbm, &cubic_hbm)).await;

        // Pass 3: (1 + tanh) * 0.5 * x
        let out_hbm = launch(gelu_pass3_kernel, (&mut ctx, &tanh_hbm, &input_hbm)).await;

        let actual: Vec<f32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.into_vec();
        let expected = reference_expected();

        println!("=== GELU PyTorch vs RNGD sim ===");
        for i in 0..5 {
            println!("  [{i}]: {:.6} | {:.6}", expected[i], actual[i]);
        }
        for i in 0..CHECK_N {
            let tol = (expected[i].abs() * 0.05_f32).max(1e-3_f32);  // 3패스 누적 오차 허용
            assert!(
                (expected[i] - actual[i]).abs() < tol,
                "mismatch at i={i}: pytorch={} rngd_sim={}", expected[i], actual[i]
            );
        }
    }
}
