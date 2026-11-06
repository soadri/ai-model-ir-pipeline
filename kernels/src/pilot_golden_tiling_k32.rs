use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_golden_ffn_gate_kernel::{
    M, K, N, pilot_golden_ffn_gate_kernel
};

fn load_bf16_bin(path: &str) -> Vec<u8> {
    std::fs::read(path).expect(&format!("failed to read {}", path))
}

fn bf16_to_f32(data: &[bf16]) -> Vec<f32> {
    data.iter().map(|v| f32::from(*v)).collect()
}

#[tokio::main]
async fn main() { println!("K-tiling test (0.6)"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn tiling_k() {
        let mut ctx = Context::acquire();

        // K=128 → K=64 × 2 tiles
        let x0_hbm = HostTensor::<bf16, m![M, K]>::from_buf(
            load_bf16_bin("data/tiling_k_x0.bin"))
            .to_hbm::<m![1], m![M, K]>(&mut ctx.pdma).await;
        let x1_hbm = HostTensor::<bf16, m![M, K]>::from_buf(
            load_bf16_bin("data/tiling_k_x1.bin"))
            .to_hbm::<m![1], m![M, K]>(&mut ctx.pdma).await;
        let w0_hbm = HostTensor::<bf16, m![N, K]>::from_buf(
            load_bf16_bin("data/tiling_k_w0.bin"))
            .to_hbm::<m![1], m![N, K]>(&mut ctx.pdma).await;
        let w1_hbm = HostTensor::<bf16, m![N, K]>::from_buf(
            load_bf16_bin("data/tiling_k_w1.bin"))
            .to_hbm::<m![1], m![N, K]>(&mut ctx.pdma).await;

        // partial_0 = x[:, :64] @ w[:, :64].T
        let p0_hbm = launch(pilot_golden_ffn_gate_kernel, (&mut ctx, &x0_hbm, &w0_hbm)).await;
        let p0: Vec<bf16> = p0_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
        let p0_f32 = bf16_to_f32(&p0);

        // partial_1 = x[:, 64:] @ w[:, 64:].T
        let p1_hbm = launch(pilot_golden_ffn_gate_kernel, (&mut ctx, &x1_hbm, &w1_hbm)).await;
        let p1: Vec<bf16> = p1_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
        let p1_f32 = bf16_to_f32(&p1);

        // accumulate (CPU)
        let out: Vec<f32> = p0_f32.iter().zip(p1_f32.iter())
            .map(|(a, b)| a + b).collect();

        let exp_bytes = load_bf16_bin("data/tiling_k_expected.bin");
        let expected: Vec<bf16> = exp_bytes.chunks(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let exp_f32 = bf16_to_f32(&expected);

        println!("=== K-tiling (K=128, tile=64, 2 tiles + add) ===");
        for i in 0..4 {
            let diff = (exp_f32[i] - out[i]).abs();
            println!("  [{i}] expected={:.4}  actual={:.4}  diff={:.4}", exp_f32[i], out[i], diff);
        }

        let mut fail = 0;
        for (&e, &a) in exp_f32.iter().zip(out.iter()) {
            let diff = (e - a).abs();
            let tol = (e.abs() * 0.1).max(2.0);
            if diff > tol { fail += 1; }
        }
        assert!(fail == 0, "K-tiling mismatch: {fail}/{}", exp_f32.len());
        println!("PASS: {} values matched (2 tiles + accumulate)", out.len());
    }
}
