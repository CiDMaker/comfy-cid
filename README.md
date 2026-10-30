# comfy-normalizedexport

ComfyUI 워크플로를 **API 포맷**으로 내보낼 때 노드 번호를 `"1".."N"`으로 다시 매기는 프론트엔드 확장입니다.

워크플로를 자주 고치다 보면 API export의 노드 번호가 계속 커지고 순서도 뒤섞입니다.
API를 소비하는 쪽 UI가 노드 번호 순서대로 입력을 노출한다면, 그 순서를 export 시점에 내가 의도한 대로 고정하고 싶어집니다.
이 확장은 export 결과물만 변환하며, **캔버스의 실제 워크플로는 절대 수정하지 않습니다.**

- 커맨드 **Export (API, normalized)** 를 메뉴 또는 단축키로 실행
- `app.graphToPrompt()`의 API 출력을 가져와 선택한 정렬 모드로 노드 순서를 정하고 `1..N`으로 재번호
- 모든 `inputs`의 링크 참조 `[nodeId, slot]`을 새 번호로 갱신
- JSON 파일로 다운로드

캔버스용 노드는 등록하지 않습니다(`NODE_CLASS_MAPPINGS`는 비어 있음).

## 설치

ComfyUI의 `custom_nodes` 폴더에 clone 한 뒤 ComfyUI를 재시작합니다.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CiDMaker/comfy-normalizedexport.git
```

Python 의존성은 없습니다. ComfyUI 프론트엔드 1.27 이상을 권장합니다(그 이하 버전에서도 동작하지만 메뉴 위치가 다릅니다. 아래 "알려진 제약" 참고).

## 사용법

1. 워크플로를 평소처럼 구성합니다.
2. 다음 중 하나로 커맨드를 실행합니다.
   - 좌측 사이드바 상단의 ComfyUI 로고 메뉴 → **File** → **Export (API, normalized)**
   - 단축키 **Ctrl+Alt+E** (macOS에서는 Cmd+Alt+E)
   - 커맨드 팔레트에서 `Export Workflow (API Format, normalized)` 검색
3. 설정된 파일명(기본 `workflow_api_normalized.json`)으로 다운로드됩니다. 완료 시 노드 수와 정렬 모드가 토스트로 표시됩니다.

결과 파일은 ComfyUI의 일반 API 포맷과 동일한 구조이므로 `/prompt` 엔드포인트에 그대로 보낼 수 있습니다.

### 설정

설정 창(톱니바퀴)의 **Normalized Export** 항목에서 바꿉니다.

| 설정 | 기본값 | 설명 |
| --- | --- | --- |
| Node order mode | `title` | 정렬 모드. `title` / `sequential` / `topological` 중 택 1 |
| Strip [n] prefix from titles | 꺼짐 | 켜면 `_meta.title`에서 앞쪽 `[n]` 표시를 제거해 내보냄 |
| Download filename | `workflow_api_normalized.json` | 다운로드 파일명. `.json`이 없으면 자동으로 붙임 |

## 정렬 모드

세 모드 모두 **결정적**입니다. 같은 입력에는 항상 같은 결과가 나옵니다.

### `title` (기본값)

노드 제목이 `[n] ...` 형식이면 그 노드들을 `n` 오름차순으로 맨 앞에 둡니다.
표시가 없는 노드는 그 뒤에 위상 정렬 순서로 이어집니다.

- `n`은 빈 번호를 허용합니다. `[10]`, `[20]`, `[30]`처럼 띄워 두면 나중에 사이에 끼워 넣기 쉽습니다.
- 같은 번호가 여러 개면 그들 사이의 순서는 위상 정렬 순서로 정합니다.
- 제목 앞부분만 봅니다. `[3] KSampler`, `[ 3 ]KSampler`는 인식하지만 `KSampler [3]`은 인식하지 않습니다.

### `sequential`

원래 노드 ID 순서입니다. 서브그래프 경로 키(`"12:37"`)는 `:`로 나눈 세그먼트별 숫자 비교를 합니다.

```
5 < 12 < 12:3 < 12:37 < 13
```

### `topological`

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

## 제목 `[n]` 규칙 예시

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

## 서브그래프 처리

서브그래프 내부 노드는 API 포맷에서 `"12:37"` 같은 **경로 키**로 평탄화됩니다.
JavaScript 객체는 정수형 키를 항상 먼저 오름차순으로 나열하고 그 뒤에 문자열 키를 붙이기 때문에, 이런 키가 섞이면 `Object.keys()` 순서가 의도와 어긋납니다.
이 확장은 경로 키도 전부 일반 정수 키 `"1".."N"`으로 바꾸고, 경로 키를 가리키던 링크 참조도 함께 갱신합니다.

## 동작 세부

- 링크 판별: 값이 길이 2 배열이고 첫 원소가 prompt에 존재하는 키일 때만 링크로 취급합니다. 그 밖의 배열 값은 손대지 않습니다.
- 입력 객체는 `structuredClone`으로 복사해 원본을 바꾸지 않습니다.
- 사이클은 정상 prompt에서는 생기지 않지만, 남는 노드가 있으면 ID 순으로 뒤에 붙입니다.
- ComfyUI가 API 출력에서 제외하는 노드(뮤트/바이패스 등)는 이 확장에서도 당연히 빠집니다.

## 개발 / 테스트

순수 로직은 `js/normalize_core.js`에 있고 ComfyUI import가 없어 단독으로 테스트할 수 있습니다.

```bash
npm test
```

Node 22 기준 `node --test "test/**/*.test.js"`를 실행합니다.

```
js/
  normalize_core.js   # 순수 로직 (정렬, 재번호, 링크 갱신)
  normalize.js        # app.registerExtension (커맨드, 메뉴, 단축키, 설정)
test/
  normalize_core.test.js
__init__.py           # NODE_CLASS_MAPPINGS = {}, WEB_DIRECTORY = "./js"
```

## 알려진 제약

- **메뉴 위치**: ComfyUI 프론트엔드 1.27.0에서 상단 메뉴 그룹 이름이 `Workflow`에서 `File`로 바뀌었습니다. 이 확장은 프론트엔드 버전을 보고 알맞은 그룹에 넣지만, 버전을 읽지 못하면 `File`에 넣습니다. 메뉴가 보이지 않으면 단축키 **Ctrl+Alt+E**나 커맨드 팔레트를 쓰세요.
- **단축키 충돌**: 다른 확장이 Ctrl+Alt+E를 먼저 잡고 있으면 등록되지 않고 경고 토스트만 뜹니다. 설정 → Keybinding에서 직접 바꿀 수 있습니다.
- **프론트엔드 API 변동**: `commands`, `menuCommands`, `settings`, `keybindings`는 프론트엔드 1.3.x에서 도입됐고 지금도 가끔 바뀝니다. 1.60 기준으로 검증했습니다.
- **파일명 질의 설정과 무관**: 코어의 "Export (API)"와 달리 `Comfy.PromptFilename` 설정(파일명 묻기)은 따르지 않고 항상 설정된 파일명을 씁니다.
- **`_meta`가 없는 노드**: 제목이 없는 노드는 `title` 모드에서 표시 없는 노드로 취급됩니다.
- **번호 자체의 의미**: 재번호는 export 결과에만 적용됩니다. 캔버스의 노드 ID는 바뀌지 않으므로, 정규화된 파일을 다시 캔버스로 불러오는 용도(워크플로 포맷)로는 쓸 수 없습니다. API 포맷은 원래 캔버스 복원용이 아닙니다.

## 라이선스

MIT
