use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_gemv_kernel::{pilot_e2e_gemv_kernel, I, J};

mod reference_data_e2e_gemv;
use reference_data_e2e_gemv::{CHECK_N, reference_matrix, reference_vector, reference_expected};

#[tokio::main]
async fn main() {
    let mut ctx = Context::acquire();
    let matrix = HostTensor::<bf16, m![I, J]>::from_buf(to_bytes(&reference_matrix()));
    let vector = HostTensor::<bf16, m![J]>::from_buf(to_bytes(&reference_vector()));
    let matrix_hbm = matrix.to_hbm(&mut ctx.pdma).await;
    let vector_hbm = vector.to_hbm(&mut ctx.pdma).await;
    let _out_hbm = launch(pilot_e2e_gemv_kernel, (&mut ctx, &matrix_hbm, &vector_hbm)).await;
    println!("Pilot E2E gemv: kernel ran");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_actual_pytorch_output() {
        let mut ctx = Context::acquire();
        let matrix = HostTensor::<bf16, m![I, J]>::from_buf(to_bytes(&reference_matrix()));
        let vector = HostTensor::<bf16, m![J]>::from_buf(to_bytes(&reference_vector()));
        let matrix_hbm = matrix.to_hbm(&mut ctx.pdma).await;
        let vector_hbm = vector.to_hbm(&mut ctx.pdma).await;

        let out_hbm = launch(pilot_e2e_gemv_kernel, (&mut ctx, &matrix_hbm, &vector_hbm)).await;
        let actual: Vec<bf16> = out_hbm.to_host::<m![I]>(&mut ctx.pdma).await.into_vec();
        let expected = reference_expected();

        println!("=== PyTorch 실제 출력 vs RNGD 시뮬레이터 출력 ===");
        for i in 0..CHECK_N {
            println!("  [{i}]: {} | {}", f32::from(expected[i]), f32::from(actual[i]));
        }
        for i in 0..CHECK_N {
            let e = f32::from(expected[i]);
            let a = f32::from(actual[i]);
            let tol = (e.abs() * 0.02_f32).max(0.5_f32);
            assert!(
                (e - a).abs() < tol,
                "mismatch at i={i}: pytorch={}  rngd_sim={}", e, a
            );
        }
    }
}
