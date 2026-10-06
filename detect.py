"""
Задание 4.1 + 5.1: детекция объектов на каждом кадре (YOLOv8).

Что делает:
  1. Берёт все кадры из папки frames/ (по порядку имён).
  2. Прогоняет каждый кадр через предобученную модель YOLOv8n (обучена на COCO, 80 классов).
  3. Сохраняет:
     - out/det_frames/<кадр>.jpg  — кадр с нарисованными рамками, классом и confidence;
     - out/det_json/<кадр>.json   — «сырые» результаты: класс, confidence, координаты рамки.

Запуск:  python detect.py
"""
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

FRAMES_DIR = Path("frames")            # откуда берём кадры
OUT_IMG = Path("out/det_frames")       # куда кладём кадры с рамками
OUT_JSON = Path("out/det_json")        # куда кладём результаты инференса
CONF = 0.35                            # порог уверенности: объекты с меньшим confidence отбрасываем

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_JSON.mkdir(parents=True, exist_ok=True)

model = YOLO("yolov8n.pt")             # при первом запуске веса скачаются сами (~6 МБ)

frames = sorted(FRAMES_DIR.glob("*.jpg"))
print(f"Кадров найдено: {len(frames)}")

total_objects = 0
for frame_path in frames:
    result = model(str(frame_path), conf=CONF, verbose=False)[0]

    # 1) Сохраняем результат в JSON
    objects = []
    for box in result.boxes:
        cls_id = int(box.cls)
        x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
        objects.append({
            "class": result.names[cls_id],
            "confidence": round(float(box.conf), 3),
            "bbox_xyxy": [x1, y1, x2, y2],
        })
    total_objects += len(objects)
    (OUT_JSON / f"{frame_path.stem}.json").write_text(
        json.dumps({"frame": frame_path.name, "objects": objects}, ensure_ascii=False, indent=2)
    )

    # 2) Рисуем рамки + подписи «класс confidence» и сохраняем кадр
    annotated = result.plot(line_width=2, font_size=12)   # возвращает картинку в формате BGR
    cv2.imwrite(str(OUT_IMG / frame_path.name), annotated)

    print(f"{frame_path.name}: {len(objects)} объектов")

print(f"Готово. Всего найдено объектов: {total_objects}")
