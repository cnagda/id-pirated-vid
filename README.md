# id-pirated-vid

# Install
```
vagrant up
```

Alternatively, run provisionVM.sh in Ubuntu

Alternatively, build opencv itself with opencv_contrib.
My build string:
```
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local \
 -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib_path>/modules -D WITH_FFMPEG=ON -D WITH_OPENMP=ON \
 -DOPENCV_ENABLE_NONFREE=ON -D BUILD_EXAMPLES=OFF -WITH_GTK=ON -D BUILD_opencv_apps=OFF -D BUILD_DOCS=OFF \
-D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF <opencv_src>
```

## Using ffmpeg

To read in an mp4 and dump all the frames through ffmpeg ->
```
ffmpeg -i Ambulance_selector2.mp4 frames%d.bmp
```

To read in an mp4 and dump frames at 24fps
```
ffmpeg -i Ambulance_selector2.mp4 -r 24 frames%d.bmp
```