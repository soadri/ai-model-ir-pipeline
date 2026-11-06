use rngd_tcp_kernel_dev::to_bytes;
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_golden_add_a16_kernel::{
    A as A16, pilot_golden_add_a16_kernel
};
use rngd_tcp_kernel_dev::kernel::pilot_golden_add_a8_kernel::{
    A as A8, pilot_golden_add_a8_kernel
};

#[tokio::main]
async fn main() { println!("reduce_sum tiling test"); }

#[cfg(test)]
mod tests {
    use super::*;

    // 트리 구조 reduce_sum: A=16 → scalar
    // step1: add(x[0:8], x[8:16])  → [8]
    // step2: add(p[0:4], p[4:8])   → [4]  (앞 4 + 뒤 4)
    // step3: add(p[0:2], p[2:4])   → [2]
    // step4: add(p[0:1], p[1:2])   → [1]  = sum

    #[tokio::test]
    async fn reduce_sum_a16() {
        let mut ctx = Context::acquire();

        // 테스트 입력: 1~16
        let x: Vec<f32> = (1..=16).map(|v| v as f32).collect();
        let expected_sum: f32 = x.iter().sum();  // = 136.0
        let expected_mean: f32 = expected_sum / 16.0;  // = 8.5

        println!("입력: {:?}", &x);
        println!("기대 sum: {}", expected_sum);

        // Step 1: x[0:8] + x[8:16] → partial[8]
        let lo: Vec<f32> = x[0..8].to_vec();
        let hi: Vec<f32> = x[8..16].to_vec();

        let lo_hbm = HostTensor::<f32, m![A8]>::from_buf(to_bytes(&lo))
            .to_hbm(&mut ctx.pdma).await;
        let hi_hbm = HostTensor::<f32, m![A8]>::from_buf(to_bytes(&hi))
            .to_hbm(&mut ctx.pdma).await;

        let p1_hbm = launch(pilot_golden_add_a8_kernel, (&mut ctx, &lo_hbm, &hi_hbm)).await;
        let p1: Vec<f32> = p1_hbm.to_host::<m![A8]>(&mut ctx.pdma).await.into_vec();
        println!("Step 1 [8]: {:?}", &p1);

        // Step 2: p1[0:4] + p1[4:8] → partial[4]
        let p1_lo: Vec<f32> = p1[0..4].to_vec();
        let p1_hi: Vec<f32> = p1[4..8].to_vec();
        // A=8 커널에 4개 데이터 → 앞 4개만 의미, 뒤 4개 0 패딩
        let mut p1_lo_pad = p1_lo.clone();
        p1_lo_pad.extend(vec![0.0f32; 4]);
        let mut p1_hi_pad = p1_hi.clone();
        p1_hi_pad.extend(vec![0.0f32; 4]);

        let p2_lo_hbm = HostTensor::<f32, m![A8]>::from_buf(to_bytes(&p1_lo_pad))
            .to_hbm(&mut ctx.pdma).await;
        let p2_hi_hbm = HostTensor::<f32, m![A8]>::from_buf(to_bytes(&p1_hi_pad))
            .to_hbm(&mut ctx.pdma).await;

        let p2_hbm = launch(pilot_golden_add_a8_kernel, (&mut ctx, &p2_lo_hbm, &p2_hi_hbm)).await;
        let p2: Vec<f32> = p2_hbm.to_host::<m![A8]>(&mut ctx.pdma).await.into_vec();
        println!("Step 2 [4]: {:?}", &p2[..4]);

        // Step 3: p2[0:2] + p2[2:4] → partial[2]
        let mut p3_lo = vec![p2[0], p2[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut p3_hi = vec![p2[2], p2[3], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let p3_lo_hbm = HostTensor::<f32, m![A8]>::from_buf(to_bytes(&p3_lo))
            .to_hbm(&mut ctx.pdma).await;
        let p3_hi_hbm = HostTensor::<f32, m![A8]>::from_buf(to_bytes(&p3_hi))
            .to_hbm(&mut ctx.pdma).await;

        let p3_hbm = launch(pilot_golden_add_a8_kernel, (&mut ctx, &p3_lo_hbm, &p3_hi_hbm)).await;
        let p3: Vec<f32> = p3_hbm.to_host::<m![A8]>(&mut ctx.pdma).await.into_vec();
        println!("Step 3 [2]: {:?}", &p3[..2]);

        // Step 4: p3[0] + p3[1] → scalar
        let mut p4_lo = vec![p3[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut p4_hi = vec![p3[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let p4_lo_hbm = HostTensor::<f32, m![A8]>::from_buf(to_bytes(&p4_lo))
            .to_hbm(&mut ctx.pdma).await;
        let p4_hi_hbm = HostTensor::<f32, m![A8]>::from_buf(to_bytes(&p4_hi))
            .to_hbm(&mut ctx.pdma).await;

        let p4_hbm = launch(pilot_golden_add_a8_kernel, (&mut ctx, &p4_lo_hbm, &p4_hi_hbm)).await;
        let p4: Vec<f32> = p4_hbm.to_host::<m![A8]>(&mut ctx.pdma).await.into_vec();

        let actual_sum = p4[0];
        let actual_mean = actual_sum / 16.0;

        println!("\n=== reduce_sum 트리 우회 검증 ===");
        println!("  기대 sum:  {}", expected_sum);
        println!("  실제 sum:  {}", actual_sum);
        println!("  기대 mean: {}", expected_mean);
        println!("  실제 mean: {}", actual_mean);
        println!("  오차:      {}", (expected_sum - actual_sum).abs());

        assert!(
            (expected_sum - actual_sum).abs() < 1e-3,
            "reduce_sum 불일치: expected={} got={}", expected_sum, actual_sum
        );
        println!("✅ PASS: reduce_sum 트리 우회 성공");
    }
}
