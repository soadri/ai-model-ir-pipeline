use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_sin_kernel::{A, pilot_e2e_sin_kernel};

mod reference_data_e2e_sin;
use reference_data_e2e_sin::{CHECK_N, reference_a, reference_expected};

#[tokio::main]
async fn main() {
    let mut ctx = Context::acquire();
    let input = HostTensor::<f32, m![A]>::from_buf(to_bytes(&reference_a()));
    let input_hbm = input.to_hbm(&mut ctx.pdma).await;
    let _out_hbm = launch(pilot_e2e_sin_kernel, (&mut ctx, &input_hbm)).await;
    println!("Pilot E2E sin: kernel ran");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_actual_pytorch_output() {
        let mut ctx = Context::acquire();
        let input = HostTensor::<f32, m![A]>::from_buf(to_bytes(&reference_a()));
        let input_hbm = input.to_hbm(&mut ctx.pdma).await;

        let out_hbm = launch(pilot_e2e_sin_kernel, (&mut ctx, &input_hbm)).await;
        let actual: Vec<f32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.into_vec();
        let expected = reference_expected();

        println!("=== PyTorch 실제 출력 vs RNGD 시뮬레이터 출력 ===");
        for i in 0..CHECK_N {
            println!("  [{i}]: {} | {}", expected[i], actual[i]);
        }
        for i in 0..CHECK_N {
            let diff = (expected[i] - actual[i]).abs();
            let tol = (0.02 * expected[i].abs()).max(1e-3);
            assert!(
                diff <= tol,
                "mismatch at i={i}: pytorch={}  rngd_sim={}  diff={} > tol={}", expected[i], actual[i], diff, tol
            );
        }
    }
}
