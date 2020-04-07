![C/C++ CI](https://github.com/cnagda/id-pirated-vid/workflows/C/C++%20CI/badge.svg?branch=master)

# id-pirated-vid

# Description

Users create a database of known videos and can compare query videos against the
database to find piracy.

# Installation

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

# Usage

After building the project, run the command line interface from the root project
folder by executing `piracy.py`

```
usage: piracy.py [-h] {ADD,QUERY} ...

Check videos for pirated content

optional arguments:
  -h, --help   show this help message and exit

type:
  {ADD,QUERY}
```

## ADD

Add video(s) to the database or recalculate the database frame/scene vocabulary.
When adding multiple videos with optional arguments, frame/scene vocabulary
will only be recalculated after the last video is added to save time.

```
usage: piracy.py ADD [-h] [-kFrame KF] [-kScene KS] [-thresholdScene TS]
                     dbPath [paths [paths ...]]

positional arguments:
  dbPath              path to database of known videos
  paths               path(s) to directories/files to add

optional arguments:
  -h, --help          show this help message and exit
  -kFrame KF          k value for frame kmeans
  -kScene KS          k value for scene kmeans
  -thresholdScene TS  threshold for inter-scene similarity
```

## QUERY

Query the database for each video at `paths` to check for piracy.

```
usage: piracy.py QUERY [-h] dbPath paths [paths ...]

positional arguments:
  dbPath      path to database of known videos
  paths       path(s) to directories/files to add

optional arguments:
  -h, --help  show this help message and exit
```

## Examples

Create a database from videos in directory `/data/videos/` and compute frame/scene
descriptors and scenes:
```
piracy.py ADD ./build/database ./data/videos/ -kFrame 2000 -kScene 200 -thresholdScene 0.15
```

Check to see if video `/data/pirated.mp4` matches any videos in the database:
```
piracy.py QUERY ./build/database ./data/pirated.mp4
```

# Contributing

# Credits

# License
