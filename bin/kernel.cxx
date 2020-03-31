#include "kernel.hpp"

#define THRESHOLD 3
#define DBPATH 1

using namespace std;

namespace fs = std::experimental::filesystem;

int isUnspecified(std::string arg) {
    return (arg == "-1");
}

int main(int argc, char** argv )
{
    if ( argc < 4 )
    {
        printf("usage: ./add dbPath vidPath threshScene \n");
        return -1;
    }

    double threshold = stod(argv[THRESHOLD]);

    auto db = database_factory(argv[DBPATH], -1, -1, threshold);

    return 0;
}
