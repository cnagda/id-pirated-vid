#include "storage.hpp"
#include "fs_compat.hpp"
#include <fstream>
#include <opencv2/core/mat.hpp>

cv::Mat oldReadMat(ifstream &fs)
{
  int rows, cols, type, channels;
    fs.read((char *)&rows, sizeof(int));     // rows
    fs.read((char *)&cols, sizeof(int));     // cols
    fs.read((char *)&type, sizeof(int));     // type
    fs.read((char *)&channels, sizeof(int)); // channels

    Mat mat(rows, cols, type);
    for (int r = 0; r < rows; r++)
    {
        fs.read((char *)(mat.data + r * cols * CV_ELEM_SIZE(type)), CV_ELEM_SIZE(type) * cols);
    }

    return mat;
}


void writeMat(const cv::Mat &mat, ofstream &fs)
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

    return {oldreadMat(fs), startIdx, endIdx};
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
        if(entry.is_directory()) {
            // video directory
            auto vid_path = entry.path();
            for(auto& item: fs::directory_iterator(entry)) {
                if(!item.is_directory()) {
                    fs::copy_file(new_db / vid_path.filename() / item.path().filename());
                }
            }

            if(fs::exists(vid_path / "frames")) {
                for(auto& frame: fs::directory_iterator(vid_path / "frames")) {
                    auto mat = oldReadMat(ifstream(frame.path()));
                    writeMat(ofstream(new_db / vid_path.filename() / "frames" / frame.path().filename()), mat);
                }
            }

            if(fs::exists(vid_path / "scenes")) {
                for(auto& scene: fs::directory_iterator(vid_path / "scenes")) {
                    auto scene = OldSceneRead(ifstream(scene.path()));
                    SceneWrite(new_db / vid_path.filename() / "scenes" / scene.path().filename(), scene);
                }
            }
        } else {
            // some metadata thing
            fs::copy_file(new_db / entry.path().filename());
        }
    }

    return 0;
}