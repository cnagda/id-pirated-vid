![C/C++ CI](https://github.com/cnagda/id-pirated-vid/workflows/C/C++%20CI/badge.svg?branch=master)

# id-pirated-vid

# Install

## Using Vagrant

First install vagrant. (Exercise left to reader)

Then from the host machine use vagrant to provision a VM

To start the VM with virtualbox default provider:
```
vagrant up
```

This will run provisioning if provisioning has not been started. To restart the vm use `vagrant reload` which will skip provisioning.

To connect to the vm, either use `vagrant ssh` or open up virtualbox/VMWare/hyperv and connect graphically. The project folder stays synced to the host machine under `/vagrant` in the ubuntu client OS. We will add an option to run our programs without debug GUI in the future.

Alternatively, run provisionVM.sh in Ubuntu to install dependencies and build opencv with SIFT

Alternatively, build opencv itself with opencv_contrib.
My build string:
```
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local \
 -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules \
-D WITH_CUDA=ON -D WITH_VA=ON -D WITH_VA_INTEL=ON -D WITH_CUBLAS=ON \
-D WITH_FFMPEG=ON -D WITH_OPENMP=ON  -DOPENCV_ENABLE_NONFREE=ON \
 -D BUILD_EXAMPLES=OFF -D WITH_GTK=ON -D BUILD_opencv_apps=OFF -D BUILD_DOCS=OFF \
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

## Building project
```
mkdir build
cd build
cmake ..
make
```

## Running Tests
```
mkdir build
cd build
cmake .. -DBUILD_TESTING=ON
make
make test
```
