import time
import cv2
import pygame
from ultralytics import YOLO
from pathlib import Path
import yaml  # pip install pyyaml

# ============== 기본 설정 ==============
MODEL_NAME     = "yolo11n.pt"     # 또는 "yolov11n.pt"
PERSON_CLSID   = 0                # COCO 'person'
THRESHOLD_SEC  = 1.5              # 연속 감지 시간
COOLDOWN_SEC   = 3.0              # 재생 후 쿨다운(초)
CONF_THRESHOLD = 0.5
IOU_THRESHOLD  = 0.5

# ============== BoT-SORT YAML 생성 ==============
from pathlib import Path
import yaml

BOTSORT_CFG = {
    "tracker_type": "botsort",

    # BoT-SORT 스코어/버퍼
    "track_high_thresh": 0.5,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.7,
    "track_buffer": 30,
    "match_thresh": 0.8,

    # BoT-SORT 게이팅
    "proximity_thresh": 0.7,
    "appearance_thresh": 0.4,

    # ★ ReID 설정(필수)
    "with_reid": True,
    "model": "auto",
    "reid_weights": "osnet_x0_25_msmt17.pt",
    "device": 0,      # CPU면 -1
    "half": True,     # GPU FP16

    # ★ GMC 방법(철자 주의!)
    "gmc_method": "sparseOptFlow",   # ← 이걸로 교체

    # (옵션)
    "fuse_score": True,
    "min_box_area": 10,
    "cmc": False
}

tracker_yaml_path = Path(__file__).parent / "botsort_local.yaml"
with open(tracker_yaml_path, "w", encoding="utf-8", newline="\n") as f:
    yaml.safe_dump(BOTSORT_CFG, f, sort_keys=False)
tracker_yaml_path = tracker_yaml_path.resolve()


# ============== 오디오 ==============
pygame.mixer.init()
pygame.mixer.music.load("voice.mp3")  # 같은 폴더
last_play_time = 0.0

# ============== 모델 ==============
model = YOLO(MODEL_NAME)

# ============== 트랙 타이머 ==============
# {track_id: {"accum": float, "last_seen": float}}
track_timer = {}

def update_track_timers(current_ids, now, frame_dt):
    for tid in current_ids:
        if tid not in track_timer:
            track_timer[tid] = {"accum": 0.0, "last_seen": now}
        else:
            track_timer[tid]["accum"] += frame_dt
            track_timer[tid]["last_seen"] = now
    for tid in list(track_timer.keys()):
        if tid not in current_ids:
            del track_timer[tid]

def should_play():
    return any(v["accum"] >= THRESHOLD_SEC for v in track_timer.values())

# ============== 추론 + 트래킹 루프 ==============
last_loop_time = time.time()
fps_smooth = None

for result in model.track(
        source=0,
        stream=True,
        tracker=str(tracker_yaml_path),  # ★ BoT-SORT yaml 경로
        classes=[PERSON_CLSID],          # 사람만
        persist=True,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        show=False
    ):
    now = time.time()
    frame_dt = now - last_loop_time
    last_loop_time = now
    fps = 1.0 / max(frame_dt, 1e-6)
    fps_smooth = fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * fps)

    # 현재 프레임에서 감지된 person의 트랙ID 수집
    current_ids = set()
    if result.boxes is not None and getattr(result.boxes, "id", None) is not None:
        ids  = result.boxes.id.cpu().tolist()
        clss = result.boxes.cls.cpu().tolist()
        for tid, c in zip(ids, clss):
            if int(c) == PERSON_CLSID:
                current_ids.add(int(tid))

    # 타이머 갱신 & 재생
    update_track_timers(current_ids, now, frame_dt)
    if should_play() and (now - last_play_time >= COOLDOWN_SEC) and not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()
        last_play_time = now

    # 시각화
    frame = result.plot()
    txt = f"FPS: {fps_smooth:.1f}"
    if current_ids:
        any_tid = next(iter(current_ids))
        acc = track_timer.get(any_tid, {}).get("accum", 0.0)
        txt += f" | TID {any_tid} accum: {acc:.2f}s"
    cv2.putText(frame, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

    cv2.imshow("YOLO BoT-SORT ReID Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ============== 종료 정리 ==============
cv2.destroyAllWindows()
pygame.mixer.music.stop()
pygame.mixer.quit()
