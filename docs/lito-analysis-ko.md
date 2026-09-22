# LiTo 레포지토리 전수조사 분석 리포트 (한국어)

> 작성일: 2026-09-22
> 작성: Claude Code (카리나 페르소나) × @bmshin94
> 대상 레포: [bmshin94/ml-lito](https://github.com/bmshin94/ml-lito) (원본: [apple/ml-lito](https://github.com/apple/ml-lito))

---

## 📎 관련 링크 모음

| 항목 | URL |
|---|---|
| 내 포크 레포 | https://github.com/bmshin94/ml-lito |
| Apple 원본 레포 | https://github.com/apple/ml-lito |
| 프로젝트 페이지 | https://apple.github.io/ml-lito/ |
| 논문 (arXiv) | https://arxiv.org/abs/2603.11047 |
| TRELLIS (서브모듈) | https://github.com/microsoft/TRELLIS |
| 사전학습 체크포인트 | https://ml-site.cdn-apple.com/models/lito/ |
| pixi (환경관리) | https://pixi.prefix.dev/ |
| MLX (Apple Silicon) | https://github.com/ml-explore/mlx |

---

## 1. 한 줄 요약

**Apple이 공개한 "사진 한 장 → 3D 오브젝트" 생성 AI 연구 코드.**
논문명 **LiTo: Surface Light Field Tokenization**, ICLR 2026 채택작.

저자: Jen-Hao Rick Chang\*, Xiaoming Zhao\*, Dorian Chan, Oncel Tuzel (\* 공동 1저자)

---

## 2. 핵심 아이디어

기존 3D 생성 모델의 한계는 **모양(geometry)** 과 **빛 반사(appearance)** 를 분리해 다뤄
결과물이 밋밋하다는 점이었다. LiTo는 발상을 바꿨다.

| 개념 | 설명 |
|---|---|
| Surface Light Field | 물체 표면의 각 지점을 "어느 방향에서 보면 어떤 색인가"로 기술하는 함수 |
| 핵심 통찰 | RGB-D 이미지는 이 Light Field를 띄엄띄엄 샘플링한 결과물이다 |
| Tokenization | 그 샘플을 **8192 토큰 × 32차원** 잠재벡터로 압축 인코딩 |
| 결과 | 하나의 latent 공간에 **모양 + 재질 + 빛 반사**가 통합됨 |

그래서 LiTo 결과물은 **프레넬 반사**(비스듬히 볼 때 반짝), **스페큘러 하이라이트** 같은
**뷰 의존(view-dependent) 효과**가 살아있다. 이것이 경쟁 모델과의 결정적 차별점.

### 성능 스펙

- **H100 GPU**: 이미지 → 3D 생성 **4.7초** (torch.compile 후, 20 Heun step + CFG)
- **M4 Max (맥북)**: 약 160초 — MLX로 Apple Silicon 네이티브 지원
- **학습 데이터**: Objaverse 84,825개 + ObjaverseXL 155,275개 = **총 240.1k**
- 생성 결과가 **입력 이미지와 좌표계 정렬**됨 (임의 방향으로 나오지 않음)

---

## 3. 폴더 전수조사

```
ml-lito/
├─ src/lito/          # 핵심 파이썬 패키지 (약 2.5만 줄)
├─ libraries/         # 직접 만들어 쓰는 의존 라이브러리 (약 5만 줄)
├─ third_party/       # 외부 비교/의존 레포 (TRELLIS 서브모듈)
├─ demos/lito/        # FastAPI 웹 데모 + Three.js 뷰어
├─ configs/           # OmegaConf YAML 설정
├─ notebooks/         # 주피터 예제 2종
├─ env/               # pixi 기반 환경설치 스크립트
├─ assets/data_splits # 학습/검증/테스트 분할 JSON
├─ scripts/train.py   # 학습 엔트리포인트
└─ docs/              # 레포 구조 문서 (+ 이 문서)
```

### 3.1 `src/lito/` — 핵심 패키지

#### `models/` — 신경망 구조

| 파일 | 줄수 | 역할 |
|---|---|---|
| `spoint_encoder.py` | 2,750 | 포인트클라우드 → 8192 latent 토큰. Perceiver 구조 + `localized_knn` / `localized_voxel` 어텐션으로 100만 점 처리 |
| `point_decoder.py` | 2,406 | 디코더 4종: `GaussianDecoderXv`(→3DGS), `MeshDecoder_v2`(→메시), `SSLatentDecoder`(→복셀), `VelocityCrossAttnDecoder`(→포인트 재샘플링) |
| `dit.py` | — | Diffusion Transformer. 이미지 조건 → latent 토큰 생성 |
| `dino.py` | 856 | DINOv2 이미지 특징 추출 → DiT 조건 |
| `layers.py` | 1,497 | `FourierEmbed`, `PluckerEmbed`, RMSNorm, SwiGLU 등 부품 |
| `vector_decoder.py` | 1,186 | 벡터 latent 디코더 |
| `pointnet_utils.py` | 1,336 | PointNet 계열 유틸 |

#### 기타 하위 패키지

- **`trainers/`** — PyTorch Lightning 모듈. `lito_trainer.py` **4,313줄**(토크나이저), `lito_dit_trainer.py` 1,567줄(생성모델)
- **`mlx/`** — Apple Silicon 전용 포팅. `convert.py` / `convert_gaussian_decoder.py`로 PyTorch ckpt → MLX 변환
- **`flow/path.py`** — Flow Matching 경로 (SiT 논문 표기법: `xt = sigma_t*x0 + alpha_t*x1`)
- **`odelibs/`** — ODE 솔버 (Euler, Heun)
- **`datasets/obj_wdset.py`** — 1,923줄. S3 tar를 WebDataset으로 스트리밍
- **`integrations/trellis/`** — Microsoft TRELLIS 메시 추출(`cube2mesh`) 연동
- **`eval_scripts/`, `script_utils/`, `preprocess_scripts/`** — 평가 지표, config 로더, LR 스케줄러

### 3.2 `libraries/plibs/` — 범용 3D 유틸 (숨은 보물)

| 파일 | 줄수 | 내용 |
|---|---|---|
| `structures.py` | **10,886** | RGBD 이미지 / 메시 데이터 구조체 |
| `data_utils.py` | 9,422 | 데이터 변환 |
| `utils.py` | 7,432 | 범용 유틸 |
| `ppoint.py` | 3,764 | `PackedPoint` — 가변길이 포인트클라우드 구조 |
| `mesh_utils.py` | 2,761 | 메시 처리 |
| `render.py` | 1,864 | gsplat / nvdiffrast 렌더링 |
| `gs_utils.py` | 1,167 | 3D Gaussian PLY 입출력 |
| `cam_utils.py` | 1,085 | 카메라 좌표계 변환 |
| `sh_utils.py` | — | 구면조화함수(SH) |

### 3.3 `libraries/blender_rendering/` — 데이터 생성

Blender 4.2로 3D 모델을 멀티뷰 RGBD로 자동 렌더링 (약 1.2만 줄).
예제 GLB(bunny, 카트 등) 포함.

### 3.4 `demos/lito/` — 웹 데모

`fastapi_lito_demo.py` (686줄) 엔드포인트:

| 메서드 | 경로 | 역할 |
|---|---|---|
| POST | `/preprocess` | 이미지 업로드 → rembg 배경제거 + 크롭 → `preprocess_id` 반환 |
| POST | `/generate-stream` | 3D 생성. **SSE**로 실시간 진행률 스트리밍 |
| GET | `/assets/{filename}` | 결과 `.ply` / `.spz` 다운로드 |
| GET | `/health` | 헬스체크 |

프론트엔드 `index_lito.html`(1,256줄)은 **Three.js 0.174 + @sparkjsdev/spark 0.1.10**
(CDN 로드, 빌드 불필요)로 가우시안 스플랫 실시간 렌더링.

### 3.5 주의: `third_party/TRELLIS`가 비어있음

```bash
# 서브모듈 미초기화 상태 → 메시 디코더 동작 불가
git submodule update --init --recursive
```

---

## 4. 설치 및 사용법

### 4.1 플랫폼별 지원 범위

| 플랫폼 | 가능 | 불가 |
|---|---|---|
| Linux + NVIDIA GPU (A100/H100/B200 검증) | 학습 / 웹데모 / 노트북 전부 | — |
| macOS Apple Silicon (M 시리즈) | 웹데모만 (MLX) | 학습, 토크나이저 노트북 |
| Windows | — | 공식 미지원 (WSL2 우회 가능성) |

VRAM은 8192 토큰 어텐션 때문에 최소 16GB, 권장 24GB 이상.

### 4.2 설치

```bash
git clone --recurse-submodules https://github.com/apple/ml-lito.git
cd ml-lito

# 이미 클론했는데 third_party/TRELLIS가 비었다면
git submodule update --init --recursive

bash env/setup.sh        # Ubuntu
bash env/setup_mac.sh    # macOS

pixi run python -c "import lito, plibs; print('OK')"
```

`env/setup.sh`가 하는 일: pixi 설치 → `pixi.lock`(557KB) 기반으로
CUDA 12.8 + PyTorch 2.9.1 + xformers + nvdiffrast + PyTorch3D 컴파일 설치 →
headless Open3D용 `libosmesa6-dev`, Blender용 X11 라이브러리 설치 →
post-install(`setup_trellis.sh`, `setup_spz.sh`, `compile_cuda_packages.sh`).

### 4.3 웹 데모 실행 (가장 쉬움)

```bash
pixi run python demos/lito/fastapi_lito_demo.py --port 8000
# → http://localhost:8000
```

플래그: `--checkpoint_url`(로컬 경로 또는 URL, 기본 `lito_dit_rgba.ckpt`), `--port`(기본 7860)

주의사항:
- 첫 실행 시 컴파일용 더미 생성 1회 자동 실행 (torch.compile / MLX 컴파일) — 수 분 소요 정상
- macOS에서 `xformers`, `flash_attn` 없음 로그는 **정상**
- 체크포인트는 `artifacts/`에 자동 다운로드 & 캐시

### 4.4 토크나이저 노트북 (Linux + NVIDIA 전용)

```bash
jupyter lab notebooks/demo_tokenizer.ipynb
```

흐름: 체크포인트 로드 → `assets/bunny.npz` 로드 → 인코딩 → 디코딩(3DGS/메시/포인트클라우드)
→ `notebooks/recon_results/`에 PLY 저장.

```python
latent = model.get_latents(xyz_w=..., rgb=..., ray_origin_direction_w=...)
gs_dicts = model.inference_estimate_gaussians(fpoint_latent=latent)
```

학습은 2^20 포인트 / 8192 토큰 기준이나, 포인트/토큰 수 변화에 강건하다고 README에 명시.

### 4.5 학습 (현실적으로 어려움)

```bash
pixi run python scripts/train.py --config configs/lito/tokenizer/lito_8k32.yaml
pixi run python scripts/train.py --config configs/lito/generator/lito_dit_8k32.yaml
```

단, config의 데이터 경로가 Apple 내부 S3(`s3://shape-tokenization/...`)라 **외부 접근 불가**.
직접 학습하려면 Blender 렌더링부터 24만 개를 자체 구축해야 함.

### 4.6 사전학습 모델

| 모델 | 용도 | 파일 |
|---|---|---|
| 토크나이저 (추천) | 포인트클라우드 → latent | `lito_new.ckpt` |
| 토크나이저 (논문판) | 논문 재현 | `lito.ckpt` |
| 이미지→3D (추천) | 사진 → 3D | `lito_dit_rgba.ckpt` |
| 이미지→3D (논문판) | 논문 재현 | `lito_dit.ckpt` |

모두 `https://ml-site.cdn-apple.com/models/lito/` 에서 **인증 없이** 다운로드 가능.

---

## 5. Q&A 정리

### Q. 플러그인? 스킬? MCP?

**셋 다 아님.** 독립 실행되는 순수 파이썬 ML 연구 레포다.

```
.claude/  없음   .mcp.json  없음   skills/  없음   plugin.json  없음
```

루트의 `CLAUDE.md`는 레포 기능이 아니라 사용자가 직접 추가한
Claude Code 프로젝트 지침 파일(커밋 `7eafc28`, PR #1).

| 구분 | 정체 | LiTo |
|---|---|---|
| 플러그인 | Claude Code 기능 확장 패키지 | ✗ |
| 스킬 | `SKILL.md`로 정의된 작업 지침서 | ✗ |
| MCP 서버 | Claude ↔ 외부 도구 연결 표준 프로토콜 | ✗ |
| LiTo | FastAPI 서버 + PyTorch 모델 + 파이썬 라이브러리 | ✓ |

단, 이미 REST API가 있으므로 **얇은 MCP 래퍼를 씌우면 Claude의 도구로 사용 가능**하다.

### Q. API 토큰이 필요한가?

**기본 사용에는 토큰이 전혀 필요 없다.** 외부 유료 API 호출 코드가 한 줄도 없고 100% 로컬 추론.

| 상황 | 인증 |
|---|---|
| 웹 데모 실행 | 불필요 |
| 체크포인트 다운로드 | 불필요 (Apple CDN 공개) |
| 노트북 토크나이저 | 불필요 |
| rembg 배경제거 | 불필요 (u2net 자동 다운로드) |
| Three.js 뷰어 | 불필요 (jsDelivr CDN) |
| 학습 시 W&B 로깅 | 선택 (`wandb_project: null`로 비활성화) |
| 학습 데이터 S3 | 접근 불가 (Apple 내부) |
| TRELLIS 일부 모델 | 경우에 따라 HuggingFace 토큰 |

비용은 전기요금 + GPU뿐. 클라우드 H100 기준 생성 1회 약 $0.005.

### Q. 왜 GitHub에서 유명한가?

1. **Apple 브랜드** — 연구를 잘 공개하지 않던 Apple이 가중치까지 공개
2. **ICLR 2026 채택** — 논문 + 코드 + 가중치 + 데이터 분할 + 웹데모 풀세트, 재현성 최상
3. **4.7초** — 수십 분 걸리던 3D 생성을 초 단위로. 연구에서 제품으로 넘어가는 임계점
4. **기술적 참신함** — "Surface Light Field를 토큰화"하여 뷰 의존 효과 재현
5. **Apple Silicon(MLX) 네이티브 지원** — 대부분의 3D AI가 CUDA 전용인데 맥북에서 동작
6. **만능 출력** — 하나의 latent에서 3DGS + 메시 + 포인트클라우드
7. **프로덕션급 코드 품질** — `pixi.lock` 완전 재현 환경, ruff + pre-commit, 구조 문서, 30KB ACKNOWLEDGEMENTS

### Q. 로컬 에이전트 구축에 도움이 되는가?

**에이전트 프레임워크로는 아니지만, 에이전트의 "도구"로는 매우 유용하다.**

도움되는 점:
1. 이미 깔끔한 도구 인터페이스(REST API)가 완성돼 있어 MCP 래퍼 약 100줄이면 연결 가능
2. 완전 로컬 → API 비용 0원, 프라이버시 보장, 온프레미스 적합
3. SSE 스트리밍 → 에이전트가 장시간 작업의 진행률을 인지·중계 가능
4. 멀티모달 에이전트의 "손" 역할 (LLM은 3D를 만들 수 없음)
5. `fastapi_lito_demo.py`의 패턴(모델 싱글톤 → 전처리 캐시 LRU 10개 → 스트리밍 응답 → 에셋 서빙)이 다른 로컬 AI 도구 서버의 템플릿이 됨

한계: GPU 상주 필요(다른 LLM과 VRAM 경쟁), 콜드 스타트(torch.compile 수 분),
데모 코드는 단일 요청 전제라 큐잉 직접 구현 필요, 가중치 연구용 라이선스.

권장 구성:

```
[로컬 LLM: Ollama/Qwen]
        ↕ MCP
[MCP 라우터]
   ├─ lito-3d-server  (3D 생성)
   ├─ filesystem      (파일 관리)
   └─ blender-mcp     (후처리 / GLB 변환)
```

### Q. React나 PHP로 만들 수 있는가?

```
프론트엔드 (React)      ✅ 100% 가능, 오히려 권장
API 게이트웨이 (PHP)    ✅ 가능 (프록시/인증/결제 역할만)
AI 추론 엔진 (Python)   ❌ 대체 불가능
```

**React** — `index_lito.html`이 이미 Three.js + spark.js 바닐라 JS라 이식이 쉽다.
권장 스택: Next.js 15 + `@react-three/fiber` + `@react-three/drei` + `@sparkjsdev/spark`
+ Zustand/TanStack Query + Tailwind + shadcn/ui.

**PHP** — 인증, 결제/크레딧, 작업 큐, 추론 서버 프록시, 결과 서빙은 가능.
단 모델 추론 자체는 불가(PyTorch/CUDA 바인딩 부재)하고, PHP-FPM에서 SSE 프록시는 버퍼링 이슈가 있다.

**Python 추론 엔진 대체 불가 이유** — PyTorch/MLX(신경망), CUDA(GPU 연산),
xformers/flash-attn(어텐션 메모리 최적화), PyTorch3D(3D 기하), nvdiffrast(미분가능 래스터라이저),
gsplat(가우시안 렌더링), Open3D, TRELLIS에 의존한다.

권장 아키텍처:

```
[React / Next.js]        업로드 UI · 진행률 · 3D 뷰어 · 갤러리
        │ HTTPS / SSE
[API 게이트웨이]          인증 · 결제 · 큐 · 레이트리밋 · 로깅  (Node.js 권장, PHP 가능)
        │ 내부망
[Python GPU 워커]        FastAPI + LiTo + PyTorch/CUDA (Docker + GPU 오토스케일)
```

---

## 6. 라이선스 — 수익화 전 필독

| 파일 | 대상 | 상업 이용 |
|---|---|---|
| `LICENSE` | **코드** | 조건부 허용 (Apple 샘플코드 라이선스, 고지 유지 등 조건 준수) |
| `LICENSE_MODEL` | **사전학습 가중치** | **금지** — "Research Purposes only" |
| `LICENSE_generated_samples` | 제공된 샘플 | 금지 — CC BY-NC-ND 4.0 |

> `LICENSE_MODEL` 22~28행 발췌:
> *"...exclusively for Research Purposes... 'Research Purposes' means non-commercial
> scientific research... does not include any commercial exploitation, product development
> or use in any commercial product or service."*

수익화 경로는 세 갈래다.

- **Route A** — 가중치를 쓰지 않고 수익화 (즉시 가능, 리스크 최소)
- **Route B** — 코드로 자체 데이터를 학습해 내 가중치를 확보 (비용 큼)
- **Route C** — Apple과 상업 라이선스 협의 (현실성 낮음)

※ 아래 내용은 법률 자문이 아니라 문서를 읽은 개발자의 해석이다.
실제 사업화 전 변호사 검토가 필요하며, 학습 데이터(Objaverse 등)의 라이선스도 별도 확인해야 한다.

---

## 7. 수익화 아이디어

### 7.1 Route A — 가중치 없이 수익화

#### A-1. 교육 콘텐츠

| 상품 | 가격대 |
|---|---|
| 인프런/유데미 강의 | 5~15만원 |
| 유튜브 시리즈 | 광고 + 멤버십 |
| 기술 뉴스레터 | 월 5천~1만원 |
| 전자책 / Notion 템플릿 | 2~5만원 |
| 기업 대상 워크숍 | 50~200만원 |

커리큘럼 초안:

```
1주차  3D 표현 방식 비교 (메시 / 복셀 / NeRF / 3DGS)
2주차  pixi로 CUDA 환경 30분 세팅
3주차  Perceiver 인코더 — 100만 포인트를 8192 토큰으로
4주차  Flow Matching + DiT 구조 해부
5주차  FastAPI + SSE + Three.js 실시간 3D 웹서비스
6주차  PyTorch → MLX 포팅 (Apple Silicon)
7주차  Blender 자동화로 나만의 3D 데이터셋 만들기
8주차  최종 프로젝트 배포
```

#### A-2. 컨설팅 / SI

국내에 "3D 생성 AI + GPU 인프라"를 동시에 아는 인력이 희소하다는 점이 해자.

| 서비스 | 단가 |
|---|---|
| 3D AI PoC 컨설팅 | 500~2,000만원 |
| 기업 맞춤 파이프라인 구축 | 3,000만원~ |
| GPU 클러스터 세팅 자문 | 시간당 15~30만원 |
| 기술 실사 리포트 | 300~1,000만원 |

타깃: 이커머스(AR 상품뷰), 게임사(에셋 파이프라인), 가구/인테리어, 부동산, 건축/제조.

#### A-3. `plibs` 기반 독립 도구 (코드 라이선스 조건 확인 필수)

- 3D 포맷 변환 SaaS (GLB ↔ PLY ↔ SPZ ↔ USDZ)
- 가우시안 스플랫 압축 서비스 (SPZ 변환)
- Blender 자동 렌더링 팜
- 3D 품질 검수 도구 (`pcd_metric_utils`, `pointersect_metric_utils`)

수익 모델: SaaS 월 $29~99 또는 일회성 $49.

#### A-4. 데이터셋 비즈니스 (라이선스 제약 없음, 마진 높음)

| 상품 | 가격 |
|---|---|
| 한국형 3D 데이터셋 (한옥, 가전, 가구/소품 RGBD 멀티뷰) | $500~5,000 |
| 산업 특화 데이터셋 (의료기기, 자동차 부품, 패션) | $2,000~20,000 |
| 렌더링 as a Service (고객 3D모델 → 학습 데이터셋 변환) | 건당 과금 |
| 주석 달린 벤치마크 (3D 생성 모델 평가용) | 구독제 |

`libraries/blender_rendering`(1.2만 줄)이 그대로 생산 설비가 된다.

#### A-5. 웹 3D 뷰어 SaaS

`index_lito.html`의 Three.js + spark.js 뷰어만 분리:

- 쇼핑몰용 3D 뷰어 위젯 (Shopify / Cafe24 앱)
- 3D 포트폴리오 호스팅
- 부동산 3D 투어 플랫폼

월 $19~199 구독. 모델 가중치를 쓰지 않으므로 라이선스 제약 없음.

### 7.2 Route B — 자체 학습 모델로 수익화

| 항목 | 규모 | 비용 |
|---|---|---|
| 데이터 렌더링 | 3~10만 오브젝트 | 200~800만원 |
| 토크나이저 학습 | H100 × 8 × 5~10일 | 1,500~4,000만원 |
| 생성모델 학습 | H100 × 8 × 5~10일 | 1,500~4,000만원 |
| 시행착오 버퍼 | ×1.5 | — |
| **합계** | | **5,000만 ~ 1.5억원** |

개인 부담은 크므로 투자 또는 정부과제(TIPS, IITP) 연계를 권장.

학습 완료 시 가능한 제품:

| 제품 | 타깃 | 가격 |
|---|---|---|
| 이미지→3D API | 개발자 | 생성당 $0.05~0.20 |
| 이커머스 3D화 플랫폼 | 쇼핑몰 | 월 $299~2,999 |
| 게임 에셋 생성 툴 | 인디 게임사 | 월 $49~499 |
| AR 필터 제작 SaaS | 마케터 | 월 $99~999 |
| 3D 프린팅 연계 | 개인 | 건당 과금 |

### 7.3 단계별 로드맵

```
Phase 1 (0~3개월) — 자본 0원, 신뢰 축적
  · LiTo 분석 기술 블로그 연재 (주 1회)
  · 한국어 데모 영상 (선점 효과)
  · MCP 서버 래퍼 오픈소스 공개
  수익: 월 0~50만원 / 얻는 것: 인지도, 포트폴리오

Phase 2 (3~9개월) — 지식 판매 + 컨설팅
  · 온라인 강의 출시
  · 기업 워크숍 / 세미나
  · 컨설팅 계약 1~2건
  수익: 월 300~1,500만원

Phase 3 (9개월~) — 제품화
  · A-4(데이터셋) 또는 A-5(뷰어 SaaS)로 자체 매출 확보
  · 확보 자금 + 투자로 Route B 진입
  수익: 월 1,000만원~ 또는 시드 투자
```

### 7.4 웹 개발 역량 보유자의 포지셔닝

```
AI 연구자   : 모델은 만들지만 제품을 못 만든다
웹 개발자   : 제품은 만들지만 AI를 모른다
둘 다 가능  : 시장의 빈틈
```

전략: **"AI 모델은 오픈소스로 가져오고, 제품은 직접 만든다."**
Phase 1의 MCP 래퍼나 A-5(뷰어 SaaS)는 웹 개발 역량만으로 즉시 착수 가능하다.

---

## 8. 다음 액션 아이템

- [ ] `git submodule update --init --recursive`로 TRELLIS 복구
- [ ] `bash env/setup.sh` (또는 `setup_mac.sh`)로 환경 구축
- [ ] `demos/lito/fastapi_lito_demo.py` 실행해 직접 체험
- [ ] `notebooks/demo_tokenizer.ipynb`로 인코딩/디코딩 흐름 파악
- [ ] LiTo MCP 서버 래퍼 프로토타입 작성
- [ ] React(Next.js) 기반 프론트엔드 재구현
- [ ] 라이선스 조건 법률 검토 후 수익화 경로 확정
