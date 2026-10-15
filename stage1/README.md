# Stage 1: MaskedKD 원인 분석 (H1 단순 약화 vs H2 student 방향 약화)

```bash
git clone https://github.com/jeehoo0507/aim-lab-test-777 MaskedKD && cd MaskedKD
export IMAGENET_ROOT=/path/to/imagenet        # train/, val/ 클래스 폴더
nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &
tail -f logs/run_all.log
```

`run_all.sh`: setup_env → prepare_in100(약 5GB memmap) → smoke_test(실패 시 중단) → core(core_s0, core_s1) → extra(extra_s0) → plot.

- 진행 확인: `tail -f logs/run_all.log`, `tail -f logs/core_s0.log`, `nvidia-smi`
- 중단: `pkill -f run_all.sh; pkill -f group_train.py`
- 재개: `nohup bash scripts/run_stage1.sh core > logs/resume.log 2>&1 &` (각 그룹은 마지막 5 epoch 체크포인트에서 `--resume`)
- 결과: `results/<group>/<student>/{metrics.jsonl,train_log.jsonl,config.json,masks_ep*.npz}`, `results/summary.csv`, `results/plots/*.png`
- 스모크 결과: `logs/smoke_report.txt`, 예상 시간: `logs/estimate.txt`
- 합성 데이터로만 스모크: `SMOKE_SYNTHETIC=1 bash scripts/smoke_test.sh` (teacher/monotone 검사는 SKIP)

## 구현상 선택 (스펙에 정해지지 않은 부분)
- 그룹 안의 모든 student는 같은 seed로 초기화(동일 init), 같은 배치·증강을 본다. stochastic depth와 random/noisy 마스크는 student별 RNG 스트림이라 그룹 구성과 무관하다 (스모크 `equiv_group`).
- CE는 원본 기본값과 같은 LabelSmoothingCE(0.1), drop_path 0.1, alpha 0.5, tau 1. `*_noce`는 loss = KD (가중치 1).
- `tau_const:T`, `tau_match`: teacher logits에만 T를 적용하고 student는 tau=1, T² 보정 없음.
- `teacher_attn_noisy`: 이미지별 z-score한 teacher CLS attention 점수에 N(0, σ²), σ = noise_init(기본 2.0)·(1−진행률).
- `rollout`: head 평균 + 0.5·I, 행 정규화 후 레이어 곱 (Abnar & Zuidema). 새 기준은 `masking.py`에 `@register`로 추가.
- probe의 "그 student 기준 마스킹 teacher": 마스킹 없는 student(`none`, `tau_*`)는 자기 attention에 원본 MaskedKD 규칙을 적용해 측정.
- 그림 1의 x축: 학습 중 probe(epoch≥2) 평균의 masked-teacher top-1 하락. 마스킹 없는 student는 0.
- epoch 0(학습 전)에도 probe를 한 번 기록한다.
- 학습률은 step 단위 warmup+cosine (warmup_lr 1e-6, min_lr 1e-5, 원본 기본값).
- teacher FLOPs: 블록 연산(토큰 수 기준) + 전체 패치에 대한 patch embedding. teacher attention이 필요한 기준은 전체 forward 비용도 포함한다.
