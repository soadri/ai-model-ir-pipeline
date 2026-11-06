use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_sub_kernel::{A, pilot_e2e_sub_kernel};

mod reference_data_e2e_sub;
use reference_data_e2e_sub::{CHECK_N, reference_a, reference_b, reference_expected};

#[tokio::main]
async fn main() {
    let mut ctx = Context::acquire();
    let lhs = HostTensor::<f32, m![A]>::from_buf(to_bytes(&reference_a()));
    let rhs = HostTensor::<f32, m![A]>::from_buf(to_bytes(&reference_b()));
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma).await;
    let _out_hbm = launch(pilot_e2e_sub_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;
    println!("Pilot E2E sub: kernel ran");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_actual_pytorch_output() {
        let mut ctx = Context::acquire();
        let lhs = HostTensor::<f32, m![A]>::from_buf(to_bytes(&reference_a()));
        let rhs = HostTensor::<f32, m![A]>::from_buf(to_bytes(&reference_b()));
        let lhs_hbm = lhs.to_hbm(&mut ctx.pdma).await;
        let rhs_hbm = rhs.to_hbm(&mut ctx.pdma).await;

        let out_hbm = launch(pilot_e2e_sub_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;
        let actual: Vec<f32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.into_vec();
        let expected = reference_expected();

        println!("=== PyTorch 실제 출력 vs RNGD 시뮬레이터 출력 ===");
        for i in 0..CHECK_N {
            println!("  [{i}]: {} | {}", f32::from(expected[i]), f32::from(actual[i]));
        }
        for i in 0..CHECK_N {
            assert!(
                (expected[i] - actual[i]).abs() < 1e-4,
                "mismatch at i={i}: pytorch={}  rngd_sim={}", expected[i], actual[i]
            );
        }
    }
}
