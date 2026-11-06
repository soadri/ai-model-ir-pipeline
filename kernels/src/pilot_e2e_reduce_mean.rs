use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_reduce_mean_kernel::{
    A, reduce_sum_sq_kernel, scale_kernel
};

mod reference_data_e2e_reduce_mean;
use reference_data_e2e_reduce_mean::{reference_input, reference_expected};

#[tokio::main]
async fn main() { println!("reduce_mean 2-pass test"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_actual_pytorch_output() {
        let mut ctx = Context::acquire();

        let input = HostTensor::<f32, m![A]>::from_buf(to_bytes(&reference_input()));
        let input_hbm = input.to_hbm(&mut ctx.pdma).await;

        let inv_n_data: Vec<f32> = vec![1.0 / A::SIZE as f32; 4];
        let inv_n = HostTensor::<f32, m![4]>::from_buf(to_bytes(&inv_n_data));
        let inv_n_hbm = inv_n.to_hbm(&mut ctx.pdma).await;

        // Pass 1
        let sum_sq_hbm = launch(reduce_sum_sq_kernel, (&mut ctx, &input_hbm)).await;

        // Pass 2
        let result_hbm = launch(scale_kernel, (&mut ctx, &sum_sq_hbm, &inv_n_hbm)).await;

        let actual: Vec<f32> = result_hbm.to_host::<m![4]>(&mut ctx.pdma).await.into_vec();
        let expected = reference_expected();

        println!("=== mean(x²) 결과 ===");
        println!("  PyTorch:  {:.6}", expected[0]);
        println!("  RNGD sim: {:.6}", actual[0]);

        let tol = (expected[0].abs() * 0.02_f32).max(1e-4_f32);
        assert!(
            (expected[0] - actual[0]).abs() < tol,
            "mismatch: pytorch={} rngd_sim={}", expected[0], actual[0]
        );
    }
}
