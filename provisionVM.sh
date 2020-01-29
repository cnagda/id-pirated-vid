#! /bin/bash
apt-get update
apt-get install -y build-essential cmake libavcodec-dev libavresample-dev libavformat-dev libswscale-dev libgtk-3-dev
mkdir /OpenCV
cd /OpenCV
git clone --depth=1 --branch 4.2.0 https://github.com/opencv/opencv.git
git clone --depth=1 --branch 4.2.0 https://github.com/opencv/opencv_contrib.git
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local \
 -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules -D WITH_FFMPEG=ON -D WITH_OPENMP=ON \
 -DOPENCV_ENABLE_NONFREE=ON -D BUILD_EXAMPLES=OFF -WITH_GTK=ON -D BUILD_opencv_apps=OFF -D BUILD_DOCS=OFF \
-D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF ../opencv
make -j8
make install