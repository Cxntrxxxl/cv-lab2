
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

FRAMES_DIR = Path("frames")
OUT_MASK = Path("out/seg_masks")
OUT_IMG = Path("out/seg_frames")
CONF = 0.35
ALPHA = 0.5         
BACKGROUND = 255     

OUT_MASK.mkdir(parents=True, exist_ok=True)
OUT_IMG.mkdir(parents=True, exist_ok=True)

model = YOLO("yolov8n-seg.pt")

hsv = np.array([[[int(179 * ((cid * 0.618) % 1)), 255, 255] for cid in range(len(model.names))]],
               dtype=np.uint8)
PALETTE = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]


def draw_legend(img, class_ids, names):
    """Рисует в левом верхнем углу легенду: цветной квадрат + название класса."""
    y = 10
    for cid in class_ids:
        color = tuple(int(c) for c in PALETTE[cid])
        cv2.rectangle(img, (10, y), (30, y + 20), color, -1)
        cv2.putText(img, names[cid], (38, y + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)          
        cv2.putText(img, names[cid], (38, y + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26


frames = sorted(FRAMES_DIR.glob("*.jpg"))
print(f"Кадров найдено: {len(frames)}")

for frame_path in frames:
    img = cv2.imread(str(frame_path))
    h, w = img.shape[:2]
    result = model(img, conf=CONF, retina_masks=True, verbose=False)[0]

    class_map = np.full((h, w), BACKGROUND, dtype=np.uint8)

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()        
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        for i in np.argsort(confs):
            class_map[masks[i] > 0.5] = classes[i]

    cv2.imwrite(str(OUT_MASK / f"{frame_path.stem}.png"), class_map)


    color_layer = img.copy()
    present = [c for c in np.unique(class_map) if c != BACKGROUND]
    for cid in present:
        color_layer[class_map == cid] = PALETTE[cid]
    overlay = cv2.addWeighted(color_layer, ALPHA, img, 1 - ALPHA, 0)
  
    for cid in present:
        contours, _ = cv2.findContours((class_map == cid).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, tuple(int(c) for c in PALETTE[cid]), 2)
    draw_legend(overlay, present, model.names)

    cv2.imwrite(str(OUT_IMG / frame_path.name), overlay)
    print(f"{frame_path.name}: классы {[model.names[c] for c in present]}")

print("Готово.")
