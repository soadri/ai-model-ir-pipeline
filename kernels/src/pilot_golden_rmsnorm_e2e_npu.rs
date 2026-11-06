use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_golden_rmsnorm_mul_kernel::{
    A as RA, pilot_golden_rmsnorm_mul_kernel
};
use rngd_tcp_kernel_dev::kernel::pilot_golden_rmsnorm_rsqrt_kernel::{
    pilot_golden_rmsnorm_rsqrt_kernel
};
use rngd_tcp_kernel_dev::kernel::pilot_golden_rmsnorm_xnorm_kernel::{
    A as AX, pilot_golden_rmsnorm_xnorm_kernel
};
use rngd_tcp_kernel_dev::kernel::pilot_golden_add_a8_kernel::{
    A as A8, pilot_golden_add_a8_kernel
};

mod reference_data_golden_rmsnorm_e2e;
use reference_data_golden_rmsnorm_e2e::{reference_input, reference_expected};

async fn reduce_sum_npu(ctx: &mut Context, x: &[f32]) -> f32 {
    let lo_hbm = HostTensor::<f32, m![A8]>::from_buf(to_bytes(&x[0..8].to_vec()))
        .to_hbm(&mut ctx.pdma).await;
    let hi_hbm = HostTensor::<f32, m![A8]>::from_buf(to_bytes(&x[8..16].to_vec()))
        .to_hbm(&mut ctx.pdma).await;
    let p1 = launch(pilot_golden_add_a8_kernel, (ctx, &lo_hbm, &hi_hbm)).await;
    let p1v: Vec<f32> = p1.to_host::<m![A8]>(&mut ctx.pdma).await.into_vec();

    let mut lo2 = p1v[0..4].to_vec(); lo2.extend(vec![0.0f32; 4]);
    let mut hi2 = p1v[4..8].to_vec(); hi2.extend(vec![0.0f32; 4]);
    let lo2_hbm = HostTensor::<f32, m![A8]>::from_buf(to_bytes(&lo2))
        .to_hbm(&mut ctx.pdma).await;
    let hi2_hbm = HostTensor::<f32, m![A8]>::from_buf(to_bytes(&hi2))
        .to_hbm(&mut ctx.pdma).await;
    let p2 = launch(pilot_golden_add_a8_kernel, (ctx, &lo2_hbm, &hi2_hbm)).await;
    let p2v: Vec<f32> = p2.to_host::<m![A8]>(&mut ctx.pdma).await.into_vec();

    let lo3 = vec![p2v[0], p2v[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let hi3 = vec![p2v[2], p2v[3], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let lo3_hbm = HostTensor::<f32, m![A8]>::from_buf(to_bytes(&lo3))
        .to_hbm(&mut ctx.pdma).await;
    let hi3_hbm = HostTensor::<f32, m![A8]>::from_buf(to_bytes(&hi3))
        .to_hbm(&mut ctx.pdma).await;
    let p3 = launch(pilot_golden_add_a8_kernel, (ctx, &lo3_hbm, &hi3_hbm)).await;
    let p3v: Vec<f32> = p3.to_host::<m![A8]>(&mut ctx.pdma).await.into_vec();

    let lo4 = vec![p3v[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let hi4 = vec![p3v[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let lo4_hbm = HostTensor::<f32, m![A8]>::from_buf(to_bytes(&lo4))
        .to_hbm(&mut ctx.pdma).await;
    let hi4_hbm = HostTensor::<f32, m![A8]>::from_buf(to_bytes(&hi4))
        .to_hbm(&mut ctx.pdma).await;
    let p4 = launch(pilot_golden_add_a8_kernel, (ctx, &lo4_hbm, &hi4_hbm)).await;
    let p4v: Vec<f32> = p4.to_host::<m![A8]>(&mut ctx.pdma).await.into_vec();

    p4v[0]
}

#[tokio::main]
async fn main() { println!("RMSNorm e2e NPU reduce test (0.6)"); }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rmsnorm_e2e_full_npu() {
        let mut ctx = Context::acquire();
        let x = reference_input();
        let expected = reference_expected();

        let x_hbm = HostTensor::<f32, m![RA]>::from_buf(to_bytes(&x))
            .to_hbm(&mut ctx.pdma).await;

        let xsq_hbm = launch(pilot_golden_rmsnorm_mul_kernel, (&mut ctx, &x_hbm)).await;
        let x_sq: Vec<f32> = xsq_hbm.to_host::<m![RA]>(&mut ctx.pdma).await.into_vec();

        let sum_xsq = reduce_sum_npu(&mut ctx, &x_sq).await;
        let mean_sq = sum_xsq / RA::SIZE as f32;
        let eps = 1e-6_f32;
        println!("Step 2 sum(x²)={:.6}  mean_sq={:.6}", sum_xsq, mean_sq);

        let rms_vec = vec![(mean_sq + eps); RA::SIZE];
        let rms_hbm = HostTensor::<f32, m![RA]>::from_buf(to_bytes(&rms_vec))
            .to_hbm(&mut ctx.pdma).await;
        let rsqrt_hbm = launch(pilot_golden_rmsnorm_rsqrt_kernel, (&mut ctx, &rms_hbm)).await;
        let rsqrt: Vec<f32> = rsqrt_hbm.to_host::<m![RA]>(&mut ctx.pdma).await.into_vec();
        println!("Step 3 rsqrt={:.6}", rsqrt[0]);

        let x_hbm2 = HostTensor::<f32, m![AX]>::from_buf(to_bytes(&x))
            .to_hbm(&mut ctx.pdma).await;
        let rsqrt_hbm2 = HostTensor::<f32, m![AX]>::from_buf(to_bytes(&rsqrt))
            .to_hbm(&mut ctx.pdma).await;
        let xnorm_hbm = launch(
            pilot_golden_rmsnorm_xnorm_kernel, (&mut ctx, &x_hbm2, &rsqrt_hbm2)
        ).await;
        let x_norm: Vec<f32> = xnorm_hbm.to_host::<m![AX]>(&mut ctx.pdma).await.into_vec();

        println!("\n=== RMSNorm e2e Full NPU (0.6) ===");
        for i in 0..RA::SIZE {
            println!("  [{i:2}] PyTorch={:7.4}  RNGD={:7.4}  diff={:.6}",
                expected[i], x_norm[i], (expected[i] - x_norm[i]).abs());
        }

        let mut fail = 0;
        for i in 0..RA::SIZE {
            let tol = (expected[i].abs() * 0.02_f32).max(1e-4_f32);
            if (expected[i] - x_norm[i]).abs() > tol { fail += 1; }
        }
        assert!(fail == 0, "mismatch: {fail}/{}", RA::SIZE);
        println!("PASS: RMSNorm Full NPU (CPU fallback 없음)");
    }
}
