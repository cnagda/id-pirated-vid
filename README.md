# id-pirated-vid

# Install
```
vagrant up
```

Alternatively, run provisionVM.sh in Ubuntu

## Using ffmpeg

To read in an mp4 and dump all the frames through ffmpeg ->
```
ffmpeg -i Ambulance_selector2.mp4 frames%d.bmp
```

To read in an mp4 and dump frames at 24fps
```
ffmpeg -i Ambulance_selector2.mp4 -r 24 frames%d.bmp
```