use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_dot_product_kernel::{A, pilot_e2e_dot_product_kernel};

mod reference_data_e2e_dot_product;
use reference_data_e2e_dot_product::{reference_a, reference_b, reference_expected};

#[tokio::main]
async fn main() {
    let mut ctx = Context::acquire();
    let lhs = HostTensor::<bf16, m![A]>::from_buf(to_bytes(&reference_a()));
    let rhs = HostTensor::<bf16, m![A]>::from_buf(to_bytes(&reference_b()));
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma).await;
    let _out_hbm = launch(pilot_e2e_dot_product_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;
    println!("Pilot E2E dot_product: kernel ran");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_bf16_reference() {
        let mut ctx = Context::acquire();
        let lhs = HostTensor::<bf16, m![A]>::from_buf(to_bytes(&reference_a()));
        let rhs = HostTensor::<bf16, m![A]>::from_buf(to_bytes(&reference_b()));
        let lhs_hbm = lhs.to_hbm(&mut ctx.pdma).await;
        let rhs_hbm = rhs.to_hbm(&mut ctx.pdma).await;

        let out_hbm = launch(pilot_e2e_dot_product_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;
        let actual_buf: Vec<bf16> = out_hbm.to_host::<m![1]>(&mut ctx.pdma).await.into_vec();
        let expected = reference_expected();

        if let Some(&actual) = actual_buf.first() {
            println!("=== 값 비교 ===");
            println!("  expected={:?} actual={:?}", expected, actual);
            let diff = (f32::from(actual) - f32::from(expected)).abs();
            let tol = (0.05 * f32::from(expected).abs()).max(1.0);
            assert!(diff <= tol, "dot_product mismatch: expected {expected:?}, actual {actual:?}, diff {diff} > tol {tol}");
        }
    }
}
