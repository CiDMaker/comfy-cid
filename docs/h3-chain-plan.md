# H3 Chain 커스텀 노드 기획서

작성 2026-11-05 · 개정 2026-11-06 · 대상 ComfyUI (gb10 192.168.0.5, 메인 PC, Modal 이미지) · comfy-cid 저장소의 `cid_nodes/h3_*.py`로 구현

## 1. 왜 만드는가

### 문제
- MiniMax H3는 한 번의 샘플링으로 4~15초만 만들 수 있고, 메모리/시간 때문에 실제로는 8~10초가 적정선. 그 이상의 연속 장면은 세그먼트를 이어 붙여야 한다.
- 기존 방식(AddGuide에 이전 컷 마지막 5프레임 가이드 + last.png ref)은 가이드 구간을 매번 다시 그려서 앞 프레임에 번인/깜빡임(4프레임마다 중복 프레임)이 생기고, 노출이 세그먼트마다 틀어진다.
- 11-04~05 테스트(`/opt/data/cache/output/h3_mask_cont/`, 그래프는 `h3_latent_chain_workflows.zip`)에서 **latent prefix + noise mask** 방식이 이를 해결함을 확인했다. 단, 결론마다 실험 조건이 다르다:

| 그래프 | 비디오 prefix | 오디오 prefix | x2 + Pass2 prefix | 이전 컷 last.png ref | 확인한 것 |
|---|---|---|---|---|---|
| I (noref) | ○ | × | × | × | 앞 구간이 다시 그려지지 않음(잔차 0.7~1.0, h264 수준), 중복 프레임 없음, 시트/플레이트만으로 됨 |
| K (cut5→cut6) | ○ | ○ (37틱) | × | × | 오디오도 이어짐(스펙트럼 거리 0.06) |
| L (K→cut7) | ○ | ○ | × | × | 3세그먼트 연쇄에서 누적 열화 없음 |
| H (x2) | ○ | × | ○ | **○** | Pass2에도 prefix를 넣어야 밝기 단차(~2)가 사라짐 |

  - "이전 영상 ref 불필요"는 드래프트(업스케일 없음)에서만, "Pass2 prefix로 밝기 단차 해소"는 last.png ref가 있고 오디오 prefix가 없는 조건에서만 확인했다.
  - 노드가 목표로 하는 조합(비디오+오디오 prefix, x2, ref 없음)은 **아직 한 번도 같이 돌지 않았다** → §5-6에서 검증한다.
- 그런데 이 방식을 코어 노드만으로 구성하면 **패스당 7~8개 노드 + 저장 4파일 + output→input 수동 업로드**가 필요해 워크플로로 관리가 불가능하다. 현재는 Python 러너(`run_mask2.py`)가 API json을 조립해서 돌리고 있다.

### 목표
- 세그먼트 하나를 ComfyUI **서브그래프 블록**으로 만들고, 블록을 N개 이어 붙이면 10초 이상의 장면이 **워크플로 하나, 큐 한 번**으로 생성되게 한다.
- 블록은 기존 `video_deno_ref` 그래프를 그대로 쓰고, 코어 노드로 안 되는 접착 부분만 커스텀 노드 3종으로 대체한다.
- ComfyUI 캐시로 "2번 블록만 시드 변경 → 2·3번만 재실행"이 되게 한다(승인/재생성 흐름).
- 세그먼트 수가 늘어도 메모리가 늘지 않게 한다.

### 만들지 않는 것
- 블록 전체를 하나의 메가 노드로 만들지 않는다. 모델 로더/LoRA/Optimize 서브그래프/업스케일 체인은 사용자가 그래프에서 계속 바꾸는 부분이라 노드 안에 묻으면 안 됨. 캐시 단위도 노드별이어야 부분 재실행이 된다.
- 컷 순서·승인·조인·검증은 노드가 아니라 그래프 구성과 스크립트(join, colormatch, Whisper, seam 잔차)가 맡는다.

## 2. 참고한 것 (gb10 ComfyUI 7c4390a6, 2026-10-31 기준으로 확인)

| 참고 | 가져온 것 |
|---|---|
| `comfy_extras/nodes_minimax_h3.py` (`_empty_av_latent`, `temporal_shape`) | AV latent 구조: nested {video [1,24,T,H/16,W/16], audio [1,32,2,Ta]}. 토큰 주기 `FRAME_PER_TOKEN=(1,4,4,4,4)` = 17k+5프레임 / 5k+2토큰, 오디오 `round(frames/24×40)`틱, 프레임당 `FRAME_RESCALE=5/3`틱 |
| `comfy/samplers.py` `KSamplerX0Inpaint`, `comfy/model_base.py` MiniMax `scale_latent_inpaint`, `comfy/ldm/minimax/model.py` | prefix가 보존되는 실제 메커니즘 세 가지: ① 샘플러가 매 스텝 마스크 구간 입력을 `scale_latent_inpaint` 결과로 바꿔 넣음(비디오는 0.999·clean + 0.001·noise, 오디오는 clean을 스케일 보정) ② 모델이 마스크 행을 cond timestep(0.999)으로 돌림 ③ 모델 출력에 마스크를 곱하고 샘플러가 출력의 마스크 구간을 `latent_image`로 덮어씀. 그래서 **출력 prefix는 입력과 정확히 같다**. 마스크는 2×2 DiT 패치 단위 amax로 풀링되고 1/256로 양자화되므로 **정확히 0/1만 쓴다**(중간값은 부분 강도) |
| 코어 `LatentCut` / `LatentConcat` (`nodes_latent.py`) | 축 선택 규칙: `t` = ndim-3, `x` = 마지막 축. video는 `t`, audio는 `x`로 잘라야 함(audio에 `t`를 쓰면 채널 축을 자름) |
| 코어 `SetLatentNoiseMask`, `SolidMask`, `MaskComposite`, KJ `CreateFadeMaskAdvanced`, `comfy.utils.reshape_mask` | 마스크 생성. 샘플러가 마스크를 trilinear로 latent 크기에 맞춰 리사이즈하므로 **시간 축 길이가 latent의 T와 같은 마스크**를 줘야 경계가 번지지 않음(공간은 프레임 전체가 같은 값이라 크기 무관) |
| `LTXVSeparateAVLatent` / `LTXVConcatAVLatent` | nested AV latent 분리/결합. Concat은 한쪽에만 마스크가 있으면 다른 쪽을 1로 채워 nested noise_mask를 만든다 |
| `SamplerCustomAdvanced` | 출력 dict는 입력 `latent.copy()`이므로 **입력의 noise_mask가 출력에 그대로 실린다**. 블록 2 이후의 P1/P2 출력과 저장 파일에는 prefix 마스크가 항상 들어 있다 |
| 코어 `SaveLatent` / `LoadLatent` | `samples.contiguous()`를 부르는데 NestedTensor에 그 메서드가 없어 nested 저장 불가(분리 후 2파일). 저장은 output/, 로드는 input/. 러너가 /upload/image로 올려서 우회 — 이 불편이 Save/Load 노드를 만드는 이유 |
| `video_deno_ref` 업스케일 서브그래프 (356:*) | `MinimaxH3LatentUpscaler3D`(356:372)는 새 dict `{"samples"}`만 반환해 video 마스크를 버리고, audio는 356:347 SolidMask 64×64 값 0 → 356:348 SetLatentNoiseMask로 전부 보존. 따라서 Pass2 앞에 video prefix를 다시 넣어야 하고, Pass2 오디오는 Pass1 오디오와 정확히 같다 |
| `run_mask2.py` (검증된 러너) | 노드가 재현해야 할 정확한 그래프: 930~938(Pass1 video), 960~967(audio), 950~957(Pass2), 940~945(save). 기반 그래프 `/opt/data/scripts/deno_ref_api_20261101e.json` = gb10 `video_deno_ref.json` 11-01 저장본의 API export. 922/923(AddGuide)은 러너의 `MODE=guide`에서만 추가되는 노드로 워크플로에는 없다 |
| 측정 결과 (`references/latent-continuation.md`) | 수치 기준: prefix 잔차 ≤1.3, 오디오 스펙트럼 거리 ≤0.1, 세그먼트 톤 차 ≤1 |

## 3. 노드 사양

### 3.1 `H3 Latent Prefix`
이전 세그먼트의 latent 끝을 새 latent 앞에 고정한다. Pass1과 Pass2 모두 같은 노드.

| 입력 | 타입 | 설명 |
|---|---|---|
| `latent` | LATENT (AV nested) | 새 세그먼트의 빈 AV latent (R2V 노드 out1) 또는 업스케일된 AV latent |
| `prev_latent` | LATENT (AV nested), **optional** | 이전 세그먼트의 샘플러 출력 AV latent (같은 패스, 같은 해상도). **`samples`만 읽는다**(dict에 실려 오는 noise_mask는 무시). 미연결이면 `latent`를 그대로 통과 → 블록 1개 = 단일 컷 워크플로 |
| `prefix_tokens` | INT, 기본 7 | video 토큰 수. 5m+2만 허용(2, 7, 12 …). 이유: 이전 latent의 끝 P토큰이 새 latent의 0..P-1에 들어가면서 (1,4,4,4,4) 주기가 맞으려면 (T_prev−P) % 5 == 0이어야 함. 아니면 에러 메시지로 올바른 값 안내 |
| `audio` | BOOLEAN, 기본 true | audio latent도 prefix. 틱 수 = `ceil(prefix_frames × 5/3)` (22프레임 → 36.67 → 37; 러너의 `round(x+0.5)`와 같은 값이지만 39프레임처럼 정확히 정수가 되는 경우 65로 맞음). 36.67이 정수가 아니라 비디오와 1/3틱(8 ms) 어긋나지만 K/L에서 문제없음이 확인된 선택 |
| (자동) | — | Pass2 처리는 입력으로 받지 않고 **자동 판정**한다: 입력 audio에 이미 noise_mask가 있으면(업스케일 서브그래프의 64×64 마스크 0) 그 마스크를 그대로 두고 오디오 prefix는 건너뜀(전부 보존되는 오디오에 prefix를 넣는 것은 no-op). 입력 마스크는 latent 모양이 아닐 수 있으므로 모양을 가정하지 않는다. 사용자가 Pass2에서 플래그를 깜빡하는 실수를 없애기 위함 |

| 출력 | 설명 |
|---|---|
| `latent` | prefix가 들어가고 noise_mask가 설정된 AV latent → 샘플러 `latent_image` |
| `prefix_frames` | INT, 조인 시 드롭할 프레임 수 (7토큰 → 22) |
| `prefix_seconds` | FLOAT, 조인 시 드롭할 오디오 초 = **`prefix_frames / 24`** (22프레임 → 0.9167). 오디오 틱 수(37틱 = 0.925 s)를 쓰면 이음새마다 A/V가 8 ms씩 어긋나 누적되므로 쓰지 않는다 |

내부 동작: separate → video `narrow(t, -P, P)` + `narrow(t, 0, T-P)` concat → video mask [T, H/16, W/16] (0..P-1 = 0.0, 나머지 1.0) → audio `narrow(x, -A, A)` + 나머지 concat → audio mask [2, Ta] (0..A-1 = 0.0) → concat AV. 입력 `latent`에 noise_mask가 있어도 video 마스크는 항상 새로 만든다. 해상도/토큰 수 불일치, prev가 더 짧은 경우, prev가 nested가 아닌 경우는 명확한 에러.

### 3.2 `H3 Save AV Latent` / `H3 Load AV Latent`
승인된 세그먼트의 latent를 파일 하나로 고정한다. ComfyUI 재시작·Modal 컨테이너 휘발·다른 호스트에서 이어가기용.

**Save** — 입력 `latent`(AV nested), `path`(STRING, 절대경로 또는 output 상대; 예 `/mnt/nas/Data/ai_studio/<project>/latents/cut6_p1.h3lat`), `overwrite`(BOOLEAN). 출력 `latent`(통과), `path`(STRING). safetensors: `video`, `audio`, `video_noise_mask?`, `audio_noise_mask?`, 메타(`frames`, `tokens`, `width`, `height`, `pass`, `seed`, `prompt_hash`, `comfy_version`).

**실행 보장은 배선과 `always_run` 입력으로 한다.** ComfyUI는 출력 노드와 그 상류만 실행하므로, 통과 노드로만 두면 §4 배선에서 마지막 블록과 뒤 블록을 mute한 승인 대기 블록의 Save는 아무도 소비하지 않아 실행되지 않는다(러너는 코어 `SaveLatent`가 출력 노드라 문제가 없었다). 그런데 Save를 무조건 OUTPUT_NODE로 두면 반대 문제가 생긴다. 스위치 349는 lazy 입력이라 선택되지 않은 358 쪽을 평가하지 않는데, 358 뒤의 P2 Save가 출력 노드면 업스케일 off에서도 Pass2가 돌아간다. 그래서:
- **P1 Save는 OUTPUT_NODE**(`OUTPUT_NODE = True`, 출력도 가짐). Pass1은 어차피 항상 돈다.
- **P2 Save는 통과 노드**로 두고, 그 통과 출력을 스위치 349의 `on_true`에 연결한다. 업스케일 on이면 Decode & Save → 스위치 → P2 Save → 358 순으로 평가되어 저장되고, off면 평가되지 않는다.
- 구현은 한 클래스에 `always_run`(BOOLEAN, 기본 true) 입력을 두는 대신 두 노드 id로 등록한다: `CiD_H3SaveAVLatent`(OUTPUT_NODE)와 동일 클래스의 통과 변형 `CiD_H3SaveAVLatentPassthrough`(OUTPUT_NODE 아님). 노드 하나의 플래그로는 OUTPUT_NODE를 바꿀 수 없기 때문이다.
출력 노드도 입력이 같으면 캐시되어 다시 쓰지 않으므로 부분 재실행에는 영향이 없다.

**저장은 옵션이 아니다**: 같은 시드로 다른 프로세스(재시작 후)에서 재실행하면 복사본이 나오지 않아(SSIM 0.94) 승인 클립의 latent는 그 실행에서만 얻을 수 있다.

**Load** — 입력 `path`(STRING). 출력 `latent`(AV nested, noise_mask는 싣지 않음), `frames`, `width`, `height`. output/·input/ 구분 없이 경로로 읽는다.

### 3.3 `H3 Decode & Save`
디코드 → mp4 저장 → **경로 문자열만** 출력. 디코드된 IMAGE 텐서를 캐시에 남기지 않는다. **OUTPUT_NODE**다(`path`를 소비하는 노드가 없으므로 출력 노드가 아니면 실행되지 않는다).

입력: `latent`(AV), `vae`, `audio_vae`, `filename_prefix`, `fps` 24, `crf`(기본 18 = 현재 SaveVideo auto와 같음), `tiled`(기본 true; 끄면 코어 VAEDecode로 한 번에 디코드), 타일 디코드 설정(512/64/64/8, 현재 그래프 369의 값 기본). 출력: `path`(STRING), `preview`(UI에 영상 표시). 현재 그래프의 369 VAEDecodeTiled + 351 VAEDecodeAudio + 354 CreateVideo(fps 24) + 357 SaveVideo를 하나로 합친 것.
근거: 블록이 끝날 때마다 VAEDecode 출력 IMAGE가 캐시에 남으면 10 s당 2.7 GB(1280×736)~11 GB(2560×1472)가 누적되고, gb10은 샘플링과 같은 121 GB 풀이라 x2 final 3블록부터 위험. 전역 캐시 플래그(LRU/none)는 승인된 앞 블록까지 재실행시키므로 부적합.

## 4. 어디에 연결하는가 (`video_deno_ref` 11-01 저장본 = `deno_ref_api_20261101e.json` 기준 노드 id)

```
R2V 365 ──out1 (빈 AV latent)──► [H3 Latent Prefix A] ──► Pass1 샘플러 362 .latent_image
                                   ▲ prev_latent = 이전 블록의 P1 출력
362 ──out1──► [H3 Save AV Latent] (cut_p1, OUTPUT_NODE) ──► 블록 출력 P1
          └─► 업스케일 서브그래프 356 (Separate 356:299 → Upscaler 356:372 / audio SolidMask0 356:347→348 → ConcatAV 356:346)
356:346 ──► [H3 Latent Prefix B] (오디오 마스크 자동 유지) ──► Pass2 샘플러 358 .latent_image
              ▲ prev_latent = 이전 블록의 P2 출력
358 ──out1──► [H3 Save AV Latent (passthrough)] (cut_p2) ──► 블록 출력 P2
                                                  └──► 스위치 349 .on_true
스위치 349 (upscale on/off: on_false=362, on_true=P2 Save 통과 출력) ──► [H3 Decode & Save, OUTPUT_NODE] (vae 340 out2, audio_vae 340 out3) ──► path
```
- 가이더 359/355의 conditioning(370 Text Encoder Unload 출력)은 손대지 않는다. 저장본에는 AddGuide가 없고, 필요하면 러너의 guide 모드처럼 AddGuide를 끼워 넣어 병용할 수 있다(prefix와 독립).
- `TextEncoderUnload 370`, Optimize 서브그래프(368), LoRA 체인(352/353/363)은 그대로.
- 기존 357 SaveVideo + 369/351/354는 Decode & Save로 대체(미리보기는 노드의 preview로).
- 362 출력 dict에는 Prefix A가 넣은 noise_mask가 실려 356으로 들어가지만, 업스케일러가 버리고 348이 오디오를 덮어쓰므로 영향 없음.

### 블록(서브그래프) `H3 Segment`
- 입력: `prev_p1`(optional), `prev_p2`(optional), ref 이미지 1~9 / ref 영상 0~3 / ref 오디오 0~3, `prompt`, `seed`, `duration`, `ratio/MP`, `upscale`
- 출력: `p1_latent`, `p2_latent`, `video_path`, `prefix_frames`, `prefix_seconds`
- 장면 = 블록 N개를 `p1→prev_p1`, `p2→prev_p2`로 직렬 연결. 블록 1은 prev 미연결 = 지금의 단일 컷과 동일.
- 승인 흐름: 뒤 블록을 mute 해두고 하나씩 풀기. 블록 k의 시드/프롬프트만 바꾸면 k 이후만 재실행(캐시). P1 Save는 출력 노드, P2 Save는 Decode & Save의 상류이므로 mute된 뒤 블록이 없어도 현재 블록의 latent는 저장된다.

### 저장 위치
- 새 워크플로 `video_deno_ref_chain`으로 저장(기존 `video_deno_ref`는 그대로). 안정되면 Prefix의 optional 입력 덕분에 하나로 합칠 수 있음.
- 노드 설치: comfy-cid에 포함(gb10·메인 PC·Modal 이미지의 `custom_nodes/comfy-cid`를 갱신). 의존성은 torch/safetensors/PyAV만(코어에 있음).

## 5. 검증 계획 (드래프트, 로컬)
1. 단위: 러너 `run_mask2.py`가 만든 API json(K/L/H)과 노드 출력 latent가 같은지(동일 입력에서 `samples` 텐서 일치, nested noise_mask 일치). 7토큰→22프레임, 37틱, 5m+2 거부, prev 미연결 통과도 단위 테스트.
2. 블록 1개, prev 미연결: 기존 `video_deno_ref`와 **한 프로세스 안에서 같은 시드로 두 그래프를 연달아 돌려** 샘플러 출력 latent를 비교. 같은 조건에서 코어 그래프 자체의 재실행 편차를 먼저 측정해 그 수준(SSIM/latent 오차)을 기준으로 삼는다. §3.2의 SSIM 0.94는 재시작을 거친 경우라 여기 기준으로 쓰지 않는다.
3. 블록 2개(cut5→cut6, seed 6602, 드래프트): 러너 결과 K와 prefix 잔차 ≤1.3, 오디오 스펙트럼 거리 ≤0.1.
4. 블록 3개 큐 한 번: 메모리 피크(`free -m` 2 s 간격)가 단일 컷 + 수백 MB 이내.
5. 블록 2 시드 변경 → 블록 1 캐시 유지(재실행 시간으로 확인). 뒤 블록을 mute한 상태에서도 블록 1·2의 Save 파일이 생기는지, 업스케일 off에서 Pass2가 돌지 않는지(P2 파일 없음, 실행 시간) 확인.
6. **x2 final 체인 2블록, 비디오+오디오 prefix, last.png ref 없음(아직 안 돌려본 조합)**: Pass2 prefix로 밝기 단차 ≤1, 오디오 스펙트럼 거리 ≤0.1, 잔차 ≤1.3. 안 되면 H와의 차이(ref, 오디오 prefix)를 하나씩 되돌려 원인을 가른다.
7. Save→재시작→Load→블록 3만 실행: 결과가 재시작 전과 동일(latent 텐서 일치).
8. 조인: `prefix_frames`/`prefix_seconds`로 자른 뒤 3세그먼트 전체의 A/V 오프셋이 누적되지 않는지(이음새마다 ±4 ms 이내).

## 6. 리스크 / 열린 점
- ComfyUI 코어 업데이트로 AV latent 구조, noise_mask 처리(`scale_latent_inpaint`, 마스크 풀링), H3 노드 출력이 바뀌면 Prefix 노드가 깨진다 → 입력 구조를 검증하고 버전을 메타에 기록.
- 서브그래프의 optional 입력과 캐시 동작은 프론트엔드 버전(gb10 1.56.3)에 따라 다를 수 있음 → gb10에서 먼저 확인.
- 세그먼트 4개 이상의 누적 드리프트는 미측정(3개까지 없음).
- 비디오+오디오 prefix와 x2의 조합은 미검증(§5-6).
- 오디오 prefix 경계가 비디오 경계와 1/3틱 어긋남. 측정상 문제없지만 원인 추적 시 염두에 둔다.
- 모델이 기억하는 건 직전 22프레임뿐이므로 장면 서사의 연속성은 블록별 프롬프트가 담당(시작 상태 한 줄 기술). 이는 노드가 해결하지 않는다.
- 오디오 prefix는 이전 컷 끝 소리를 이어가므로 컷 경계는 대사/호흡 사이에 둔다(프롬프트 규칙).

## 개정 내역 (2026-11-06, 2차)
- §3.2 P1 Save만 OUTPUT_NODE, P2 Save는 통과 변형을 스위치 `on_true`에 연결(업스케일 off에서 Pass2가 강제 실행되는 것을 막음). 두 노드 id로 등록.
- §3.3 Decode & Save는 OUTPUT_NODE임을 명시.
- §3.1 `keep_audio_mask` 입력을 없애고 자동 판정으로. 오디오 틱 식을 `ceil`로.
- §4 배선, §5-5 검증 항목 갱신.

## 개정 내역 (2026-11-06, 1차)
- §3.2 Save를 OUTPUT_NODE로 변경. 통과 노드면 마지막·승인 대기 블록의 latent가 저장되지 않음.
- §3.1 `prefix_seconds`를 `prefix_frames/24`로 변경(A/V 8 ms 누적 방지). 오디오 틱 식을 러너와 같게 명시. `prev_latent`는 `samples`만 읽고, `keep_audio_mask`면 오디오 prefix를 건너뜀.
- §2 denoise_mask 보존 메커니즘을 코어 코드대로 세 단계로 다시 적음. 마스크는 정확히 0/1. `SamplerCustomAdvanced` 출력에 noise_mask가 실리는 점, 업스케일러가 새 dict를 반환하는 점 추가.
- §1 실험별 조건 표 추가. "ref 불필요"와 "Pass2 prefix 효과"는 서로 다른 조건에서 확인된 것이며 전체 조합은 미검증임을 명시.
- §4 922/923은 러너 전용 노드이며 저장본에 AddGuide가 없음을 반영. 디코드 경로 노드(369/351/354) 명시.
- §5-2를 한 프로세스 내 비교로 바꾸고 §3.2의 SSIM 0.94와 조건을 구분. §5-6을 미검증 조합으로 교체. §5-8 조인 A/V 검사 추가.
