use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::custom_relu_small_kernel::{
    A, custom_relu_small_kernel
};

fn load_f32_bin(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect(&format!("failed to read {}", path));
    bytes.chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[tokio::main]
async fn main() { println!("custom relu verify"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn custom_relu() {
        let mut ctx = Device::new(custom_relu_small_kernel.topology()).unwrap();
        let input = load_f32_bin("data/custom_relu_input.bin");
        let expected = load_f32_bin("data/custom_relu_expected.bin");

        let x_hbm = HostTensor::<f32, m![A]>::from_buf(to_bytes(&input))
            .to_hbm(&mut ctx.pdma).await.unwrap();
        let out_hbm = launch(custom_relu_small_kernel, (&mut ctx, &x_hbm)).await.unwrap();
        let result: Vec<f32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.unwrap().into_vec();

        println!("=== custom relu ===");
        let mut max_diff: f32 = 0.0;
        for i in 0..A::SIZE {
            let diff = (expected[i] - result[i]).abs();
            if diff > max_diff { max_diff = diff; }
            if i < 4 {
                println!("  [{}] expected={:.6} actual={:.6} diff={:.6}", i, expected[i], result[i], diff);
            }
        }
        assert!(max_diff < 0.01, "relu mismatch: max_diff={max_diff}");
        println!("PASS: relu max_diff={:.6}", max_diff);
    }
}
