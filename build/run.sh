export DISPLAY=:4
rm out.mpg
rm output*
cmake ..
make
vglrun ./demo &
timeout 30s ffmpeg -f x11grab -framerate 25 -video_size 1920x1080 -i :4 out.mpg &
