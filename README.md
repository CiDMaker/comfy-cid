# comfy-cid

ComfyUI 개인용 확장 모음입니다. 두 부분으로 되어 있습니다.

- **Export (API, normalized)**: API 포맷 export의 노드 번호를 `"1".."N"`으로 다시 매기는 프론트엔드 커맨드
- **CiD 노드**: 통합 모델 로더, 길이→프레임 수, 종횡비/메가픽셀 해상도 계산, H3 keyframe 크기 변환

## 설치

ComfyUI의 `custom_nodes` 폴더에 clone 한 뒤 ComfyUI를 재시작합니다.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CiDMaker/comfy-cid.git
```

Python 의존성은 없습니다(ComfyUI 자체가 제공하는 것만 씁니다). ComfyUI 프론트엔드 1.27 이상을 권장합니다(그 이하 버전에서도 동작하지만 메뉴 위치가 다릅니다. 아래 "알려진 제약" 참고).

## CiD 노드

노드 검색에서 `[CiD]`로 찾을 수 있고, 카테고리는 `CiD/loaders`, `CiD/video`, `CiD/image`입니다.

### Model Loader (UNET + CLIP + 2 VAE)

Load Diffusion Model, Load CLIP, Load VAE 2개를 한 노드로 합친 것입니다. 비디오 VAE와 오디오 VAE가 따로 있는 모델(MiniMax H3 등)에 맞춰 두었습니다.

- 입력: `unet_name`, `weight_dtype`, `clip_name`, `clip_type`, `clip_device`, `video_vae_name`, `audio_vae_name`
- 출력: `MODEL`, `CLIP`, `VIDEO_VAE`, `AUDIO_VAE`

선택지와 동작은 코어 로더와 같습니다. ComfyUI가 새 CLIP 타입이나 가중치 dtype을 추가하면 이 노드에도 바로 나타납니다.

**입력 하나만 바꿔도 네 모델이 모두 다시 로드됩니다.** 한 조각만 자주 바꾸는 워크플로라면 그 부분은 코어 로더를 따로 두는 편이 낫습니다.

### Video Frames from Duration

초 단위 길이와 fps로 프레임 수(`frames`)를 냅니다. `mode`로 계산 규칙을 고릅니다.

| mode | 규칙 |
| --- | --- |
| `default` (기본값) | `floor(duration × fps + 1)`. 일반 비디오 모델용 |
| `h3` | `round(duration × fps)`를 5 이상으로 잡은 뒤 17k+5(5, 22, 39, 56, …) 중 가장 가까운 위의 값으로 올림. MiniMax H3용 |

| duration | fps | default | h3 |
| --- | --- | --- | --- |
| 5.0 | 24 | 121 | 124 |
| 1.0 | 24 | 25 | 39 |
| 0.0 | 24 | 1 | 5 |

`h3`의 상수 17, 5, 5는 advanced 입력 `frame_step`, `frame_offset`, `min_frames`로 바꿀 수 있습니다. `default`에서는 쓰이지 않습니다.

### Size Parameters

종횡비와 메가픽셀로 가로세로를 계산하고, `scale_by`를 곱한 업스케일 가로세로도 함께 냅니다. 이미지와 비디오 모두에 씁니다.

| 입력 | 뜻 | 기본값 |
| --- | --- | --- |
| `ratio_width`, `ratio_height` | 종횡비. 9:16이면 9, 16 | 1, 1 |
| `megapixels` | 목표 픽셀 수. 1 = 1024×1024 | 1.0 |
| `multiple_of` | 모든 변을 이 값의 배수로 맞춤 | 16 |
| `scale_by` | 업스케일 배율 | 2.0 |

출력은 `width`, `height`, `upscaled_width`, `upscaled_height`, `scale_by`, `multiple_of`입니다. `scale_by`와 `multiple_of`는 입력값이 그대로 나오므로 이 노드 하나를 제어판처럼 쓸 수 있습니다.

기본값이면 1024×1024, 업스케일 2048×2048이 나옵니다. 9:16에 0.5 MP, multiple_of 32로 하면 544×960, 업스케일 1088×1920입니다. `width × height`는 대략 `megapixels × 1024 × 1024`이고 각 변은 `multiple_of`의 배수로 반올림됩니다.

### Resize H3 Keyframes to Latent

MiniMax H3의 FLF(first/last frame) conditioning을 다른 크기의 latent에 맞게 고칩니다. Pass1 conditioning을 업스케일된 latent로 2차 샘플링할 때 씁니다.

MiniMax H3 Image to Video가 내는 conditioning은 텍스트 임베딩과 `minimax_keyframes`(first/last 이미지를 캔버스 크기로 맞춰 VAE 인코딩한 latent)로 되어 있습니다. DiT는 keyframe latent를 샘플링할 latent의 공간 그리드에 맞춰 패킹하므로 **keyframe latent의 가로세로는 latent와 정확히 같아야** 합니다. 크기에 묶인 부분은 이것뿐이라, 2차 패스를 위해 텍스트 인코더(Qwen3-VL-32B)를 다시 돌릴 필요가 없습니다. 이 노드는 keyframe만 바꾸고 나머지는 그대로 둡니다.

| 입력 | 뜻 |
| --- | --- |
| `conditioning` | Pass1의 MiniMax H3 Image to Video 출력 |
| `latent` | 다음 패스가 샘플링할 latent(업스케일 결과). 가로세로와 프레임 수를 여기서 읽음 |
| `vae` (optional) | 비디오 VAE. `first_frame`/`last_frame`을 연결했을 때 필요 |
| `first_frame` (optional) | latent 크기로 늘려(stretch) 다시 인코딩, 0번 프레임 keyframe |
| `last_frame` (optional) | latent 크기로 center crop 해 다시 인코딩, 마지막 프레임 keyframe |

출력은 `conditioning` 하나입니다.

- 가로세로는 `latent × 16`, 프레임 수는 latent의 T에서 계산하므로 별도의 width/height 입력이 없습니다. 업스케일러가 실제로 만든 크기와 어긋날 일이 없습니다.
- 연결한 `first_frame`/`last_frame`은 코어 노드와 같은 규칙(first는 stretch, last는 center crop)으로 리사이즈한 뒤 VAE로 다시 인코딩합니다. 기존 keyframe이 있으면 그 자리에 들어가고, 없으면 추가됩니다.
- 그 외 keyframe(예: Add Guide for MiniMax H3로 넣은 앵커)은 인덱스와 오디오를 유지한 채 비디오 latent만 bilinear로 보간합니다. 이미지와 VAE를 안 연결하면 first/last도 이 방식으로 처리됩니다. 화질은 다시 인코딩하는 쪽이 낫습니다.
- latent 길이와 맞지 않는 keyframe(프레임 인덱스가 범위 밖)이 있으면 에러를 냅니다. conditioning과 latent가 다른 길이로 만들어진 경우입니다.

Pass1 텍스트 임베딩에는 원래 캔버스 크기 이미지의 vision 토큰이 들어 있습니다. 업스케일 크기로 MiniMax H3 Image to Video를 다시 돌리면 vision 토큰 수가 달라지는데, 이 노드는 그 차이를 재현하지 않습니다. 결과에 차이가 느껴지면 기존 방식과 비교해 보세요.

## Export (API, normalized)

워크플로를 **API 포맷**으로 내보낼 때 노드 번호를 `"1".."N"`으로 다시 매기는 프론트엔드 커맨드입니다.

워크플로를 자주 고치다 보면 API export의 노드 번호가 계속 커지고 순서도 뒤섞입니다.
API를 소비하는 쪽 UI가 노드 번호 순서대로 입력을 노출한다면, 그 순서를 export 시점에 내가 의도한 대로 고정하고 싶어집니다.
이 확장은 export 결과물만 변환하며, **캔버스의 실제 워크플로는 절대 수정하지 않습니다.**

- 커맨드 **Export (API, normalized)** 를 메뉴 또는 단축키로 실행
- `app.graphToPrompt()`의 API 출력을 가져와 선택한 정렬 모드로 노드 순서를 정하고 `1..N`으로 재번호
- 모든 `inputs`의 링크 참조 `[nodeId, slot]`을 새 번호로 갱신
- JSON 파일로 다운로드

### 사용법

1. 워크플로를 평소처럼 구성합니다.
2. 다음 중 하나로 커맨드를 실행합니다.
   - 좌측 사이드바 상단의 ComfyUI 로고 메뉴 → **File** → **Export (API, normalized)**
   - 단축키 **Ctrl+Alt+E** (macOS에서는 Cmd+Alt+E)
   - 커맨드 팔레트에서 `Export Workflow (API Format, normalized)` 검색
3. 설정된 파일명(기본 `workflow_api_normalized.json`)으로 다운로드됩니다. 완료 시 노드 수와 정렬 모드가 토스트로 표시됩니다.

결과 파일은 ComfyUI의 일반 API 포맷과 동일한 구조이므로 `/prompt` 엔드포인트에 그대로 보낼 수 있습니다.

#### 설정

설정 창(톱니바퀴)의 **Normalized Export** 항목에서 바꿉니다.

| 설정 | 기본값 | 설명 |
| --- | --- | --- |
| Node order mode | `title` | 정렬 모드. `title` / `sequential` / `topological` 중 택 1 |
| Strip [n] prefix from titles | 꺼짐 | 켜면 `_meta.title`에서 앞쪽 `[n]` 표시를 제거해 내보냄 |
| Download filename | `workflow_api_normalized.json` | 다운로드 파일명. `.json`이 없으면 자동으로 붙임 |

### 정렬 모드

세 모드 모두 **결정적**입니다. 같은 입력에는 항상 같은 결과가 나옵니다.

#### `title` (기본값)

노드 제목이 `[n] ...` 형식이면 그 노드들을 `n` 오름차순으로 맨 앞에 둡니다.
표시가 없는 노드는 그 뒤에 위상 정렬 순서로 이어집니다.

- `n`은 빈 번호를 허용합니다. `[10]`, `[20]`, `[30]`처럼 띄워 두면 나중에 사이에 끼워 넣기 쉽습니다.
- 같은 번호가 여러 개면 그들 사이의 순서는 위상 정렬 순서로 정합니다.
- 제목 앞부분만 봅니다. `[3] KSampler`, `[ 3 ]KSampler`는 인식하지만 `KSampler [3]`은 인식하지 않습니다.

#### `sequential`

원래 노드 ID 순서입니다. 서브그래프 경로 키(`"12:37"`)는 `:`로 나눈 세그먼트별 숫자 비교를 합니다.

```
5 < 12 < 12:3 < 12:37 < 13
```

#### `topological`

Kahn 알고리즘으로 실행 의존 순서를 따릅니다. 동시에 준비된 노드가 여러 개면 원래 ID 순(세그먼트 비교)으로 결정합니다.

ComfyUI 기본 txt2img 워크플로(노드 3~9)를 예로 들면:

| 원래 ID | 노드 | 새 ID |
| --- | --- | --- |
| 4 | Load Checkpoint | 1 |
| 5 | Empty Latent Image | 2 |
| 6 | CLIP Text Encode (positive) | 3 |
| 7 | CLIP Text Encode (negative) | 4 |
| 3 | KSampler | 5 |
| 8 | VAE Decode | 6 |
| 9 | Save Image | 7 |

### 제목 `[n]` 규칙 예시

노드 제목을 더블클릭해 다음처럼 바꿔 둡니다(모드 `title`).

| 노드 제목 | 결과 순서 |
| --- | --- |
| `[1] Save Image` | 1번 |
| `[10] Empty Latent` | 2번 |
| `[10] Negative Prompt` | 3번 (같은 번호 10은 위상 순서로: Latent가 Negative보다 먼저 준비됨) |
| `[20] Positive Prompt` | 4번 |
| `Load Checkpoint` (표시 없음) | 5번부터 위상 순서 |
| `KSampler` (표시 없음) | 6번 |
| `VAE Decode` (표시 없음) | 7번 |

즉 결과 파일의 `"1"`은 Save Image, `"2"`는 Empty Latent... 가 됩니다.
소비하는 쪽에서 "노드 번호 순으로 입력 노출"을 하면 Save Image의 파일명 접두사가 가장 위에, 그 다음 Latent 크기, 네거티브, 포지티브 순으로 뜹니다.

**Strip [n] prefix from titles**를 켜면 `_meta.title`이 `Save Image`, `Empty Latent`처럼 깔끔해집니다. 끄면 `[1] Save Image` 그대로 나갑니다(소비하는 쪽에서 번호를 보고 싶을 때).

### 서브그래프 처리

서브그래프 내부 노드는 API 포맷에서 `"12:37"` 같은 **경로 키**로 평탄화됩니다.
JavaScript 객체는 정수형 키를 항상 먼저 오름차순으로 나열하고 그 뒤에 문자열 키를 붙이기 때문에, 이런 키가 섞이면 `Object.keys()` 순서가 의도와 어긋납니다.
이 확장은 경로 키도 전부 일반 정수 키 `"1".."N"`으로 바꾸고, 경로 키를 가리키던 링크 참조도 함께 갱신합니다.

### 동작 세부

- 링크 판별: 값이 길이 2 배열이고 첫 원소가 prompt에 존재하는 키일 때만 링크로 취급합니다. 그 밖의 배열 값은 손대지 않습니다.
- 입력 객체는 `structuredClone`으로 복사해 원본을 바꾸지 않습니다.
- 사이클은 정상 prompt에서는 생기지 않지만, 남는 노드가 있으면 ID 순으로 뒤에 붙입니다.
- ComfyUI가 API 출력에서 제외하는 노드(뮤트/바이패스 등)는 이 확장에서도 당연히 빠집니다.

## 개발 / 테스트

export 로직은 `js/normalize_core.js`에, 노드의 계산은 `cid_nodes/*.py`의 모듈 함수에 있어 ComfyUI 없이 테스트할 수 있습니다.

```bash
npm test        # JS (Node 22 기준)
npm run test:py # Python unittest
```

```
js/
  normalize_core.js   # 순수 로직 (정렬, 재번호, 링크 갱신)
  normalize.js        # app.registerExtension (커맨드, 메뉴, 단축키, 설정)
cid_nodes/
  __init__.py         # NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS
  loader.py           # CiD_ModelLoader
  video_frames.py     # CiD_VideoFrames
  size_params.py      # CiD_SizeParams
test/
  normalize_core.test.js
  test_cid_nodes.py
__init__.py           # cid_nodes의 매핑을 재노출, WEB_DIRECTORY = "./js"
```

## 알려진 제약 (Export)

- **메뉴 위치**: ComfyUI 프론트엔드 1.27.0에서 상단 메뉴 그룹 이름이 `Workflow`에서 `File`로 바뀌었습니다. 이 확장은 프론트엔드 버전을 보고 알맞은 그룹에 넣지만, 버전을 읽지 못하면 `File`에 넣습니다. 메뉴가 보이지 않으면 단축키 **Ctrl+Alt+E**나 커맨드 팔레트를 쓰세요.
- **단축키 충돌**: 다른 확장이 Ctrl+Alt+E를 먼저 잡고 있으면 등록되지 않고 경고 토스트만 뜹니다. 설정 → Keybinding에서 직접 바꿀 수 있습니다.
- **프론트엔드 API 변동**: `commands`, `menuCommands`, `settings`, `keybindings`는 프론트엔드 1.3.x에서 도입됐고 지금도 가끔 바뀝니다. 1.60 기준으로 검증했습니다.
- **파일명 질의 설정과 무관**: 코어의 "Export (API)"와 달리 `Comfy.PromptFilename` 설정(파일명 묻기)은 따르지 않고 항상 설정된 파일명을 씁니다.
- **`_meta`가 없는 노드**: 제목이 없는 노드는 `title` 모드에서 표시 없는 노드로 취급됩니다.
- **번호 자체의 의미**: 재번호는 export 결과에만 적용됩니다. 캔버스의 노드 ID는 바뀌지 않으므로, 정규화된 파일을 다시 캔버스로 불러오는 용도(워크플로 포맷)로는 쓸 수 없습니다. API 포맷은 원래 캔버스 복원용이 아닙니다.

## 라이선스

MIT
