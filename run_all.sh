#!/usr/bin/env bash
# Весь конвейер одной командой:  bash run_all.sh input.mp4
# (перед запуском: conda activate cv-lab2)
set -e

VIDEO="${1:-input.mp4}"   # путь к видео (по умолчанию input.mp4)
FPS=3                     # сколько кадров в секунду вырезаем (задание: 2–5)
GIF_WIDTH=640             # ширина GIF в пикселях

# --- Задание 3: извлечение кадров ---------------------------------------
rm -rf frames out && mkdir -p frames   # чистим результаты прошлого запуска
ffmpeg -hide_banner -loglevel error -i "$VIDEO" -vf "fps=$FPS" -q:v 2 frames/frame_%04d.jpg
echo "Извлечено кадров: $(ls frames | wc -l)"

# --- Задание 4–6: инференс и визуализация --------------------------------
python detect.py
python segment.py

# --- Задания 5.2 и 6.2: сборка GIF ---------------------------------------
# palettegen/paletteuse — строим палитру под конкретное видео, чтобы GIF не был «грязным»
make_gif () {
  ffmpeg -hide_banner -loglevel error -y -framerate $FPS -i "$1/frame_%04d.jpg" \
    -vf "scale=$GIF_WIDTH:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" \
    -loop 0 "$2"
  echo "Собран $2 ($(du -h "$2" | cut -f1))"
}
make_gif out/det_frames out/detection.gif
make_gif out/seg_frames out/segmentation.gif
