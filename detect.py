
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

FRAMES_DIR = Path("frames")          
OUT_IMG = Path("out/det_frames")       
OUT_JSON = Path("out/det_json")        
CONF = 0.35                            

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_JSON.mkdir(parents=True, exist_ok=True)

model = YOLO("yolov8n.pt")            
frames = sorted(FRAMES_DIR.glob("*.jpg"))
print(f"Кадров найдено: {len(frames)}")

total_objects = 0
for frame_path in frames:
    result = model(str(frame_path), conf=CONF, verbose=False)[0]

  
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


    annotated = result.plot(line_width=2, font_size=12)  
    cv2.imwrite(str(OUT_IMG / frame_path.name), annotated)

    print(f"{frame_path.name}: {len(objects)} объектов")

print(f"Готово. Всего найдено объектов: {total_objects}")
