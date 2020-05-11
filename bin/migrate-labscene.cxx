#include "storage.hpp"
#include "fs_compat.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/core/mat.hpp>

using namespace std;

cv::Mat oldReadMat(ifstream &fs)
{
    int rows, cols, type, channels;
    fs.read((char *)&rows, sizeof(int));     // rows
    fs.read((char *)&cols, sizeof(int));     // cols
    fs.read((char *)&type, sizeof(int));     // type
    fs.read((char *)&channels, sizeof(int)); // channels

    cv::Mat mat(rows, cols, type);
    for (int r = 0; r < rows; r++)
    {
        fs.read((char *)(mat.data + r * cols * CV_ELEM_SIZE(type)), CV_ELEM_SIZE(type) * cols);
    }

    return mat;
}


void newWriteMat(const cv::Mat &mat, ofstream &fs)
{
    int type = mat.type(), dims = mat.dims;
    fs.write((char *)&type, sizeof(int));     // type
    fs.write((char *)&dims, sizeof(int));     // type

    if(dims <= 2) {
        fs.write((char *)&mat.rows, sizeof(int)); // rows
        fs.write((char *)&mat.cols, sizeof(int)); // cols
    } else {
        fs.write((char*)mat.size.p, dims * sizeof(int));
    }

    if (mat.isContinuous())
    {
        fs.write(mat.ptr<char>(), mat.total()*mat.elemSize());
    } else {
        for(int r = 0; r < mat.rows; r++) {
            fs.write(mat.ptr<char>(r), mat.cols * mat.elemSize());
        }
    }
}

SerializableScene OldSceneRead(const std::string &filename)
{
    size_t startIdx = 0, endIdx = 0;

    ifstream fs(filename, fstream::binary);
    fs.read((char *)&startIdx, sizeof(startIdx));
    fs.read((char *)&endIdx, sizeof(endIdx));

    return {oldReadMat(fs), startIdx, endIdx};
}


int main(int argc, char* argv[]) {
    if(argc < 3) {
        std::cout << "usage: labscene <db_path> <new_dbpath>" << std::endl;
        return -1;
    }

    fs::path old_db(argv[1]);
    fs::path new_db(argv[2]);

    if(!fs::exists(old_db)) {
        std::cerr << "old database path does not exist" << std::endl;
        return -1;
    }

    fs::create_directories(new_db);

    for(auto& entry: fs::directory_iterator(old_db)) {
        if(fs::is_directory(entry.path())) {
            // video directory
            auto vid_path = entry.path();
            fs::create_directories(new_db / vid_path.filename());
            for(auto& item: fs::directory_iterator(entry)) {
                if(!fs::is_directory(item.path())) {
                    fs::copy_file(item.path(), new_db / vid_path.filename() / item.path().filename());
                }
            }

            if(fs::exists(vid_path / "frames")) {
                fs::create_directories(new_db / vid_path.filename() / "frames");
                for(auto& frame: fs::directory_iterator(vid_path / "frames")) {
                    ifstream stream(frame.path());
                    auto mat = oldReadMat(stream);
                    ofstream writer(new_db / vid_path.filename() / "frames" / frame.path().filename());
                    newWriteMat(mat, writer);
                }
            }

            if(fs::exists(vid_path / "scenes")) {
                fs::create_directories(new_db / vid_path.filename() / "scenes");
                for(auto& scene: fs::directory_iterator(vid_path / "scenes")) {
                    auto s = OldSceneRead(scene.path());
                    SceneWrite(new_db / vid_path.filename() / "scenes" / scene.path().filename(), s);
                }
            }
        } else {
            // some metadata thing
            fs::copy_file(entry.path(), new_db / entry.path().filename());
        }
    }

    return 0;
}