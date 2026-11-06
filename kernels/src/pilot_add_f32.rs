use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_add_f32_kernel::{A, pilot_add_f32_kernel};

mod reference_data_f32;
use reference_data_f32::{CHECK_N, reference_a, reference_b, reference_expected};

#[tokio::main]
async fn main() {
    let mut ctx = Context::acquire();
    let lhs = HostTensor::<f32, m![A]>::from_buf(to_bytes(&reference_a()));
    let rhs = HostTensor::<f32, m![A]>::from_buf(to_bytes(&reference_b()));
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma).await;
    let _out_hbm = launch(pilot_add_f32_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;
    println!("Pilot Add f32 (Fp cluster, AddF): kernel ran");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_python_reference_f32() {
        let mut ctx = Context::acquire();
        let lhs = HostTensor::<f32, m![A]>::from_buf(to_bytes(&reference_a()));
        let rhs = HostTensor::<f32, m![A]>::from_buf(to_bytes(&reference_b()));
        let lhs_hbm = lhs.to_hbm(&mut ctx.pdma).await;
        let rhs_hbm = rhs.to_hbm(&mut ctx.pdma).await;

        let out_hbm = launch(pilot_add_f32_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;
        let actual: Vec<f32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.into_vec();
        let expected = reference_expected();

        println!("=== f32 값 비교 ===");
        for i in 0..CHECK_N {
            println!("  [{i}]: {} | {}", expected[i], actual[i]);
        }

        for i in 0..CHECK_N {
            assert!(
                (expected[i] - actual[i]).abs() < 1e-5,
                "mismatch at i={i}: expected {}, actual {}", expected[i], actual[i]
            );
        }
    }
}
