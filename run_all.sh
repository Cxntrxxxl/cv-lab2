#!/usr/bin/env bash
set -e

VIDEO="${1:-input.mp4}"   
FPS=3                    
GIF_WIDTH=640             

rm -rf frames out && mkdir -p frames   
ffmpeg -hide_banner -loglevel error -i "$VIDEO" -vf "fps=$FPS" -q:v 2 frames/frame_%04d.jpg
echo "Извлечено кадров: $(ls frames | wc -l)"

python detect.py
python segment.py

make_gif () {
  ffmpeg -hide_banner -loglevel error -y -framerate $FPS -i "$1/frame_%04d.jpg" \
    -vf "scale=$GIF_WIDTH:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" \
    -loop 0 "$2"
  echo "Собран $2 ($(du -h "$2" | cut -f1))"
}
make_gif out/det_frames out/detection.gif
make_gif out/seg_frames out/segmentation.gif
