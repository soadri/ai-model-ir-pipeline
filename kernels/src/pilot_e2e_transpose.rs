use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_transpose_kernel::{M, N, pilot_e2e_transpose_kernel};

mod reference_data_e2e_transpose;
use reference_data_e2e_transpose::{CHECK_N, reference_input, reference_expected};

#[tokio::main]
async fn main() {
    let mut ctx = Context::acquire();
    let input = HostTensor::<f32, m![M, N]>::from_buf(to_bytes(&reference_input()));
    let input_hbm = input.to_hbm(&mut ctx.pdma).await;
    let _out = launch(pilot_e2e_transpose_kernel, (&mut ctx, &input_hbm)).await;
    println!("done");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_actual_pytorch_output() {
        let mut ctx = Context::acquire();
        let input = HostTensor::<f32, m![M, N]>::from_buf(to_bytes(&reference_input()));
        let input_hbm = input.to_hbm(&mut ctx.pdma).await;

        let out_hbm = launch(pilot_e2e_transpose_kernel, (&mut ctx, &input_hbm)).await;
        let actual: Vec<f32> = out_hbm.to_host::<m![N, M]>(&mut ctx.pdma).await.into_vec();
        let expected = reference_expected();

        println!("=== PyTorch 실제 출력 vs RNGD 시뮬레이터 출력 ===");
        for i in 0..CHECK_N {
            println!("  [{i}]: {} | {}", expected[i], actual[i]);
        }
        for i in 0..CHECK_N {
            let tol = (expected[i].abs() * 0.02_f32).max(1e-4_f32);
            assert!(
                (expected[i] - actual[i]).abs() < tol,
                "mismatch at i={i}: pytorch={}  rngd_sim={}", expected[i], actual[i]
            );
        }
    }
}
