use rngd_tcp_kernel_dev::kernel::gemm_m16_k256_kernel::{
    M, K, N, gemm_m16_k256_kernel
};
use furiosa_opt_std::prelude::*;

fn load_bf16_bin(path: &str) -> Vec<u8> {
    std::fs::read(path).expect(&format!("failed to read {}", path))
}
fn bf16_to_f32(data: &[bf16]) -> Vec<f32> { data.iter().map(|v| f32::from(*v)).collect() }
fn bf16_vec_from_bin(path: &str) -> Vec<bf16> {
    load_bf16_bin(path).chunks(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect()
}

#[tokio::main]
async fn main() { println!("M=16 K=256 test"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_m16k256_gemm() {
        let mut ctx = Context::acquire();
        let t0 = std::time::Instant::now();

        let a_hbm = HostTensor::<bf16, m![M, K]>::from_buf(
            load_bf16_bin("data/test_m16k256_a.bin")).to_hbm(&mut ctx.pdma).await;
        let b_hbm = HostTensor::<bf16, m![N, K]>::from_buf(
            load_bf16_bin("data/test_m16k256_b.bin")).to_hbm(&mut ctx.pdma).await;

        let out_hbm = launch(gemm_m16_k256_kernel, (&mut ctx, &a_hbm, &b_hbm)).await;
        let out: Vec<bf16> = out_hbm.to_host::<m![M, N]>(&mut ctx.pdma).await.into_vec();
        let out_f32 = bf16_to_f32(&out);
        let exp_f32 = bf16_to_f32(&bf16_vec_from_bin("data/test_m16k256_expected.bin"));

        let elapsed = t0.elapsed().as_secs_f64();
        let mut max_diff: f32 = 0.0;
        let mut ref_max: f32 = 0.0;
        for i in 0..16*512 {
            let d = (exp_f32[i] - out_f32[i]).abs();
            if d > max_diff { max_diff = d; }
            if exp_f32[i].abs() > ref_max { ref_max = exp_f32[i].abs(); }
        }
        let rel_err = if ref_max > 0.0 { max_diff / ref_max * 100.0 } else { 0.0 };

        println!("\n=== M=16 K=256 GEMM Result ===");
        println!("  M=16, K=256, N=512");
        println!("  elapsed: {:.2}s", elapsed);
        println!("  max_diff={:.6}  ref_max={:.4}  rel_err={:.2}%", max_diff, ref_max, rel_err);
        for i in 0..8 {
            println!("  [{}] exp={:.4} act={:.4} diff={:.6}", i, exp_f32[i], out_f32[i], (exp_f32[i]-out_f32[i]).abs());
        }
        println!("  M=512 대비 연산량: {:.1}배 감소", 512.0/16.0);
        let pass = max_diff < 2.0;
        println!("  → {}", if pass { "PASS" } else { "FAIL" });
        assert!(pass, "M=16 K=256 FAIL: max_diff={max_diff}");
    }
}
