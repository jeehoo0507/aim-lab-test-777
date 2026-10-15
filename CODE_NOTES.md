# CODE_NOTES — 원본 MaskedKD 코드 동작 (코드에서 직접 확인)

기준 커밋: `96d052d` (effl-lab/MaskedKD master)

## 1. student attention 추출 위치
- `models_student.py` `VisionTransformer.forward_features`: 블록 루프에서 **마지막 블록만** `blk(x, return_attention=True)`로 attention을 받는다.
- `Attention.forward`가 반환하는 `attn`은 `softmax(q·kᵀ·scale)` 이후, `attn_drop` 적용 **후** 텐서(기본 `attn_drop_rate=0`이라 동일). shape `(B, heads, N+1, N+1)`.
- `losses.py` `DistillationLoss.forward`: `attn.mean(dim=1)[:, 0, 1:]` → **헤드 평균**, **CLS 쿼리 행(0)**, **패치 키 열(1:)** 만 사용 → `(B, N)` 점수.
- student forward는 `engine.py`에서 `torch.cuda.amp.autocast()` 안에서 실행되므로 점수는 AMP 정밀도로 계산된다.

## 2. `--len_num_keep`
- `main.py`: `--len_num_keep` (기본 196) → `DistillationLoss(len_num_keep=...)`.
- `torch.topk(scores, len_num_keep).indices` → **attention이 높은 상위 K개 패치를 유지**. 인덱스는 값 내림차순(정렬되지 않은 공간 순서)이며, 그대로 gather 인덱스로 쓰인다.
- 마스크 선택은 `--maskedkd` 여부와 무관하게 항상 계산되고, `maskedkd=False`면 teacher가 무시한다.

## 3. teacher에서 마스킹 토큰 제거 (`models_teacher.py` `forward_features(x, len_keep, maskedkd)`)
1. `patch_embed` → CLS 토큰 concat → **`x + pos_embed`** (전체 197 토큰에 pos embed를 먼저 더함)
2. `maskedkd=True`면 CLS를 떼어 두고, 패치 토큰에서 `torch.gather(x, 1, len_keep[..., None].repeat(1,1,D))`로 K개만 남긴 뒤 CLS를 다시 앞에 붙임 → 길이 K+1.
3. 이후 `pos_drop` → 모든 블록 → `norm` → `x[:, 0]` → `head`.
- 즉 pos embed는 **제거 전에** 더해지므로 남은 토큰은 원래 위치 정보를 유지한다. 토큰 순서는 topk 순서이나, ViT는 순서 등변이고 CLS 출력은 순서에 불변.
- teacher Block은 attention을 반환하지 않는다(teacher attention은 원본에 없음 → stage1에서 동일 모듈로 루프를 재구현해 추출, 동치성 확인).

## 4. distillation loss (`losses.py`)
- `base_criterion`: `main.py`에서 mixup/cutmix가 켜지면 `SoftTargetCrossEntropy`, 아니면 `--smoothing`(기본 0.1)>0이면 `LabelSmoothingCrossEntropy(0.1)`. stage1은 mixup 없음 → **LabelSmoothingCE(0.1)**.
- soft: `KLDivLoss(reduction='batchmean')(log_softmax(s/T), softmax(t/T)) * T*T`.
- 결합: `loss = base_loss * (1 - alpha) + distillation_loss * alpha`, 기본 `--distillation-alpha 0.5`, `--distillation-tau 1.0`.

## 5. 마스크 선택의 gradient
- `topk(...).indices`는 정수 인덱스라 미분 불가 → **마스크 선택으로 gradient가 흐르지 않는다**.
- teacher forward는 `torch.no_grad()` 안. KD/CE gradient는 student logits로만 흐른다.
- attention 점수 자체는 detach되지 않지만, 인덱스로만 쓰이므로 무관.

## 기타 확인 사항
- student: `models_student.__dict__[args.model](num_classes, drop_rate, drop_path_rate)`, scratch, 기본 `--drop-path 0.1`.
- optimizer: timm `create_optimizer` (AdamW, 1-D 파라미터와 `pos_embed`/`cls_token` weight decay 제외), lr = `lr * batch / 512`, warmup 5 epoch(기본), `--min-lr 1e-5`, `--warmup-lr 1e-6`, grad clip 없음.
- teacher: `models_teacher.deit_small_patch16_224(pretrained=True)` (`--teacher_model deit_small`), `eval()`.

## stage1에서의 재사용
- student 모델: `models_student.deit_tiny_patch16_224(img_size=160, num_classes=100, drop_path_rate=0.1)` 그대로.
- teacher 모델: `models_teacher.deit_small_patch16_224(img_size=160)` 그대로, 체크포인트의 pos embed를 14×14→10×10 bicubic 보간, `head`를 ImageNet-100 행만 잘라낸 `Linear`로 교체.
- 마스킹 forward: 원본 `teacher(x, len_keep, True)`를 직접 호출.
- `student_attn` 점수/선택: `attn.mean(1)[:,0,1:]` + `topk` (원본과 동일 식), KD loss: 원본과 동일 식. 스모크 테스트 1에서 원본 `losses.DistillationLoss`와 수치 비교.
