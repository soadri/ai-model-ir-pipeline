use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_golden_silu_small_kernel::{
    A, pilot_golden_silu_small_kernel
};

fn load_f32_bin(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect(&format!("failed to read {}", path));
    bytes.chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[tokio::main]
async fn main() { println!("custom silu verify"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn custom_silu() {
        let mut ctx = Context::acquire();
        let input = load_f32_bin("data/custom_silu_input.bin");
        let expected = load_f32_bin("data/custom_silu_expected.bin");

        let x_hbm = HostTensor::<f32, m![A]>::from_buf(to_bytes(&input))
            .to_hbm(&mut ctx.pdma).await;
        let out_hbm = launch(pilot_golden_silu_small_kernel, (&mut ctx, &x_hbm)).await;
        let result: Vec<f32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.into_vec();

        println!("=== custom silu ===");
        let mut max_diff: f32 = 0.0;
        for i in 0..A::SIZE {
            let diff = (expected[i] - result[i]).abs();
            if diff > max_diff { max_diff = diff; }
            if i < 4 {
                println!("  [{}] expected={:.6} actual={:.6} diff={:.6}", i, expected[i], result[i], diff);
            }
        }
        assert!(max_diff < 0.01, "silu mismatch: max_diff={max_diff}");
        println!("PASS: silu max_diff={:.6}", max_diff);
    }
}
