use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_batch_gemm_kernel::{M, K, N, pilot_e2e_batch_gemm_kernel};

fn load_bf16_bin(path: &str) -> Vec<u8> {
    std::fs::read(path).expect(&format!("failed to read {}", path))
}

#[tokio::main]
async fn main() {
    let mut ctx = Context::acquire();
    let a = HostTensor::<bf16, m![M, K]>::from_buf(load_bf16_bin("data/e2e_batch_gemm_a.bin"));
    let b = HostTensor::<bf16, m![N, K]>::from_buf(load_bf16_bin("data/e2e_batch_gemm_b.bin"));
    let a_hbm = a.to_hbm(&mut ctx.pdma).await;
    let b_hbm = b.to_hbm(&mut ctx.pdma).await;
    let _out = launch(pilot_e2e_batch_gemm_kernel, (&mut ctx, &a_hbm, &b_hbm)).await;
    println!("pilot_e2e_batch_gemm: kernel ran");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_bf16_reference() {
        let mut ctx = Context::acquire();
        let a = HostTensor::<bf16, m![M, K]>::from_buf(load_bf16_bin("data/e2e_batch_gemm_a.bin"));
        let b = HostTensor::<bf16, m![N, K]>::from_buf(load_bf16_bin("data/e2e_batch_gemm_b.bin"));
        let a_hbm = a.to_hbm(&mut ctx.pdma).await;
        let b_hbm = b.to_hbm(&mut ctx.pdma).await;

        let out_hbm = launch(pilot_e2e_batch_gemm_kernel, (&mut ctx, &a_hbm, &b_hbm)).await;
        let actual: Vec<bf16> = out_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();

        let expected_bytes = load_bf16_bin("data/e2e_batch_gemm_expected.bin");
        let expected: Vec<bf16> = expected_bytes.chunks(2).map(|c| {
            bf16::from_bits(u16::from_le_bytes([c[0], c[1]]))
        }).collect();

        println!("=== 값 비교 (앞 4개) ===");
        for i in 0..4.min(actual.len()) {
            println!("  [{i}]: expected={:?} actual={:?}", expected[i], actual[i]);
        }

        for (idx, (&e, &a)) in expected.iter().zip(&actual).enumerate() {
            let diff = (f32::from(a) - f32::from(e)).abs();
            let tol = (0.05 * f32::from(e).abs()).max(1.0);
            assert!(diff <= tol, "mismatch at idx={idx}: expected {e:?}, actual {a:?}, diff={diff}");
        }
        println!("PASS: {} values matched", actual.len());
    }
}
