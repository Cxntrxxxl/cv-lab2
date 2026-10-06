"""
Задание 4.2 + 6.1: семантическая сегментация каждого кадра (YOLOv8-seg).

Модель YOLOv8n-seg выдаёт маску для каждого найденного объекта. Чтобы получить
именно СЕМАНТИЧЕСКУЮ сегментацию (каждому пикселю — класс), маски всех объектов
одного класса объединяются в одну карту классов кадра.

Что сохраняется:
  - out/seg_masks/<кадр>.png  — карта классов: значение пикселя = id класса COCO, 255 = фон;
  - out/seg_frames/<кадр>.jpg — кадр с полупрозрачными цветными масками и легендой.

Запуск:  python segment.py
"""
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

FRAMES_DIR = Path("frames")
OUT_MASK = Path("out/seg_masks")
OUT_IMG = Path("out/seg_frames")
CONF = 0.35
ALPHA = 0.5          # прозрачность цветной маски (0 — не видно, 1 — полностью закрашено)
BACKGROUND = 255     # «класс» фона в карте классов

OUT_MASK.mkdir(parents=True, exist_ok=True)
OUT_IMG.mkdir(parents=True, exist_ok=True)

model = YOLO("yolov8n-seg.pt")

# Постоянный яркий цвет для каждого класса (одинаковый на всех кадрах, чтобы GIF не «мигал»).
# Оттенки разносятся по кругу через «золотое сечение», чтобы соседние классы не были похожи.
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
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)          # чёрная обводка
        cv2.putText(img, names[cid], (38, y + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)    # белый текст
        y += 26


frames = sorted(FRAMES_DIR.glob("*.jpg"))
print(f"Кадров найдено: {len(frames)}")

for frame_path in frames:
    img = cv2.imread(str(frame_path))
    h, w = img.shape[:2]
    result = model(img, conf=CONF, retina_masks=True, verbose=False)[0]

    # Карта классов: сначала всё — фон
    class_map = np.full((h, w), BACKGROUND, dtype=np.uint8)

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()          # (N, H, W), значения 0/1
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        # Идём от менее уверенных к более уверенным, чтобы при перекрытии
        # пиксель достался объекту с большим confidence
        for i in np.argsort(confs):
            class_map[masks[i] > 0.5] = classes[i]

    cv2.imwrite(str(OUT_MASK / f"{frame_path.stem}.png"), class_map)

    # Цветное наложение
    color_layer = img.copy()
    present = [c for c in np.unique(class_map) if c != BACKGROUND]
    for cid in present:
        color_layer[class_map == cid] = PALETTE[cid]
    overlay = cv2.addWeighted(color_layer, ALPHA, img, 1 - ALPHA, 0)
    # Контур по краю каждой маски — чтобы границы были видны даже на светлых объектах
    for cid in present:
        contours, _ = cv2.findContours((class_map == cid).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, tuple(int(c) for c in PALETTE[cid]), 2)
    draw_legend(overlay, present, model.names)

    cv2.imwrite(str(OUT_IMG / frame_path.name), overlay)
    print(f"{frame_path.name}: классы {[model.names[c] for c in present]}")

print("Готово.")
