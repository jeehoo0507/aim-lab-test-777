# PATCHES — 원본 코드에 대한 변경 내역

**원본 파일(`main.py`, `engine.py`, `losses.py`, `models_student.py`, `models_teacher.py` 등)은 수정하지 않았다.**

확인 환경: PyTorch 2.11 (CUDA 13.0 wheel), timm 1.0.x, Python 3.11.

| 항목 | 상황 | 처리 |
|---|---|---|
| `timm.models.registry` / `timm.models.helpers` / `timm.models.layers` import | timm 1.x에서 deprecated 경로지만 shim이 남아 있어 그대로 동작 (FutureWarning만 발생) | 패치 없음 |
| `@register_model` 이름 중복 (`deit_*`, `vit_*`) | `models_student.py`와 `models_teacher.py`가 timm 내장 이름을 덮어써 `UserWarning: Overwriting ... in registry` 발생. stage1은 registry를 거치지 않고 모듈 함수를 직접 호출하므로 영향 없음 | `stage1/common.py`에서 import 시 경고만 억제 |
| `torch.cuda.amp.autocast()` (`engine.py`) | 최신 PyTorch에서 deprecated | stage1은 `engine.py`를 쓰지 않음 (`torch.autocast("cuda", dtype=bfloat16)` 사용). 패치 없음 |
| 160 해상도 teacher | `deit_small_patch16_224(pretrained=True)`는 224 pos embed(197)를 고정 로드 | 모델 정의는 그대로 두고 `img_size=160`으로 생성한 뒤, 체크포인트의 pos embed를 14×14→10×10 bicubic 보간해 로드 (`stage1/common.py: build_teacher`) |
| teacher head | 1000 클래스 | `teacher.head`를 ImageNet-100 행만 복사한 `nn.Linear(384, 100)`로 교체 (모듈 교체일 뿐 forward 코드는 원본 그대로) |
| teacher attention | 원본 teacher `Block`은 attention을 반환하지 않음 | 모델 정의를 바꾸지 않고, 같은 서브모듈로 forward를 단계별 재실행해 attention을 꺼내는 함수(`teacher_full(need_attn=True)`)를 추가. 원본 forward와의 logits 일치는 스모크 테스트 `equiv_tattn`에서 확인 |
