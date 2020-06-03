export DISPLAY=:4
rm out.mpg
cmake ..
make
vglrun ./demo &
timeout 60s ffmpeg -f x11grab -framerate 25 -video_size 1920x1080 -i :4 out.mpg &
