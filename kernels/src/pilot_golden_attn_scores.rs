use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_golden_attn_qkt_kernel::{
    I, J, pilot_golden_attn_qkt_kernel
};

fn load_bf16_bin(path: &str) -> Vec<u8> {
    std::fs::read(path).expect(&format!("failed to read {}", path))
}

#[tokio::main]
async fn main() { println!("golden attn scores test (0.6)"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_pytorch_golden_reference() {
        let mut ctx = Context::acquire();

        let m_hbm = HostTensor::<bf16, m![I, J]>::from_buf(
            load_bf16_bin("data/golden_attn_scores_matrix.bin")).to_hbm(&mut ctx.pdma).await;
        let v_hbm = HostTensor::<bf16, m![J]>::from_buf(
            load_bf16_bin("data/golden_attn_scores_vector.bin")).to_hbm(&mut ctx.pdma).await;

        let out_hbm = launch(pilot_golden_attn_qkt_kernel, (&mut ctx, &m_hbm, &v_hbm)).await;
        let actual: Vec<bf16> = out_hbm.to_host::<m![I]>(&mut ctx.pdma).await.into_vec();

        let exp_bytes = load_bf16_bin("data/golden_attn_scores_expected.bin");
        let expected: Vec<bf16> = exp_bytes.chunks(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect();

        println!("=== Attn Scores (QKt) — Golden Reference (0.6) ===");
        for i in 0..4.min(actual.len()) {
            println!("  [{i}] expected={:?}  actual={:?}", expected[i], actual[i]);
        }

        let mut fail = 0;
        for (idx, (&e, &a)) in expected.iter().zip(&actual).enumerate() {
            let diff = (f32::from(a) - f32::from(e)).abs();
            let tol = (0.05 * f32::from(e).abs()).max(1.0);
            if diff > tol { fail += 1; }
        }
        assert!(fail == 0, "mismatch: {fail}/{}", actual.len());
        println!("PASS: {} values matched", actual.len());
    }
}
