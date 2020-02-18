#include "kmeans2.hpp"
#include <limits>
#include <omp.h>
#include <cmath>

typedef std::vector<std::vector<float>> mymatrix;

// TODO constrained random
mymatrix randMatrix(int rows, int cols){
    mymatrix retval(rows, std::vector<float>(cols));

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            retval[i][j] = ((float)rand())/RAND_MAX;
        }
    }

    return retval;
}

std::vector<float> randVector(int size){
    std::vector<float> retval(size);

    for(int i = 0; i < retval.size(); i++){
        retval[i] = ((float)rand())/RAND_MAX;
    }

    return retval;
}

mymatrix zeroMatrix(int rows, int cols){
    return std::vector<std::vector<float>>(rows, std::vector<float>(cols, 0));
}

mymatrix cvtomy(cv::Mat input){
    mymatrix retval(input.rows, std::vector<float>(input.cols));

    for(int i = 0; i < input.rows; i++){
        for(int j = 0; j < input.cols; j++){
            retval[i][j] = input.at<float>(i, j);
        }
    }

    return retval;
}

cv::Mat mytocv(mymatrix input){
    cv::Mat retval(input.size(), input[0].size(), CV_32F);

    for(int i = 0; i < input.size(); i++){
        for(int j = 0; j < input[0].size(); j++){
            retval.at<float>(i, j) = input[i][j];
        }
    }

    return retval;
}

float distance2(std::vector<float>& p1, std::vector<float>& p2){
    if(p1.size() != p2.size()){
        std::cerr << "SIZES DON'T MATCH in distance2: ";
        std::cerr << p1.size() << " x " << p2.size() << std::endl;
    }

    float retval = 0;

    for(int i = 0; i < p1.size(); i++){
        float a = p1[i], b = p2[i];
        retval += (a - b) * (a - b);
    }

    return retval;
}

float distance(std::vector<float>& p1, std::vector<float>& p2){
    if(p1.size() != p2.size()){
        std::cerr << "SIZES DON'T MATCH in distance2: ";
        std::cerr << p1.size() << " x " << p2.size() << std::endl;
    }

    float retval = 0;

    for(int i = 0; i < p1.size(); i++){
        float a = p1[i], b = p2[i];
        retval += (a - b) * (a - b);
    }

    return sqrt(retval);
}

void printMat(mymatrix i){
    for(auto& a : i){
        for(auto& b : a){
            std::cout << b << " ";
        }
        std::cout << std::endl;
    }
}

void matrix2csv(mymatrix i, std::string fname){
    std::ofstream of(fname);
    
    for(auto& a : i){
        for(auto& b : a){
            of << b << ",";
        }
        of << std::endl;
    }
}

cv::Mat kmeans2(cv::Mat input, int K, int attempts, float halt, int epochs){
    mymatrix data = cvtomy(input); 
    std::cout << "Data, " << data.size() << " by " << data[0].size() << std::endl;
  
    mymatrix best_centers;
    float lowest_error = std::numeric_limits<float>::max();


    for(int i = 0; i < attempts; i++){
        mymatrix centers = randMatrix(K, data[0].size());
        std::cout << "Random centers, " << centers.size() << " by " << centers[0].size() << std::endl;

        float prev_error = std::numeric_limits<float>::max();        
        float error = prev_error;

        // halting stuff
        float min_initial_distance = std::numeric_limits<float>::max();
        float max_distance = 0;
        int run = 0;

        do{
            prev_error = error;
            error = 0;
            max_distance = 0;

            int procs = omp_get_num_procs();
            //int procs = 1;

            std::vector<mymatrix> alltotals(procs, zeroMatrix(K, data[0].size()));
            std::vector<std::vector<int>> allcounts(procs, std::vector<int>(K));

            #pragma omp parallel for num_threads(procs)
            for(int j = 0; j < data.size(); j++){ // for each point
                auto& totals = alltotals[omp_get_thread_num()];
                auto& counts = allcounts[omp_get_thread_num()];

                int closest = -1;
                float dclose = std::numeric_limits<float>::max();

                for(int k = 0; k < K; k++){ // find closest center
                    float temp;
                    if((temp = distance2(data[j], centers[k])) < dclose){
                        dclose = temp;
                        closest = k;
                    }
                }

                error += dclose;
                for(int k = 0; k < totals[0].size(); k++){// add to total
                    totals[closest][k] += data[j][k];
                }
                counts[closest]++;
            }

            // merge totals and centers
            mymatrix totals = zeroMatrix(K, data[0].size());
            std::vector<int> counts(K, 0);

            for(int p = 0; p < procs; p++){
                //std::cout << "alltotals[" << p << "]" << std::endl;
                //printMat(alltotals[p]);
                //std::cout << "allcounts[" << p << "]" << std::endl;
                //for(auto& au : allcounts[p]){
                //    std::cout << au << " ";
                //}
                //std::cout << std::endl;

                for(int pi = 0; pi < alltotals[p].size(); pi++){
                    for(int pj = 0; pj < alltotals[p][0].size(); pj++){
                        totals[pi][pj] += alltotals[p][pi][pj];
                    }
                }

                for(int pi = 0; pi < K; pi++){
                    counts[pi] += allcounts[p][pi];
                }
            }


            //std::cout << "totals" << std::endl;
            //printMat(totals);
            //std::cout << "counts" << std::endl;
            //for(auto& au : counts){
            //    std::cout << au << " ";
            //}
            //std::cout << std::endl << std::endl << std::endl;

            // find new centers
            for(int j = 0; j < K; j++){
                if(counts[j]){
                    for(auto& a : totals[j]){
                        a /= counts[j];
                    }
                    auto temp = distance(centers[j], totals[j]);
                    if(!run && temp < min_initial_distance){
                        min_initial_distance = temp;
                    }
                    if(temp > max_distance){
                        max_distance = temp;
                    }

                    centers[j] = totals[j];
                }
                else{
                    centers[j] = randVector(data[0].size());
                }
            }

            std::cout << "Attempt " << i << ", Error is " << error << std::endl;
            std::cout << "max distance / min initial = " << max_distance/min_initial_distance << "/" << halt << std::endl;

        }while((prev_error - error > 0.0005) && (++run < epochs) && (max_distance/min_initial_distance > halt));

        std::cout << "Done with all attempts, error: " << error << ", lowest: " << lowest_error << std::endl;
        std::cout << "Took " << run << " runs" << std::endl;
        if(error < lowest_error){
            lowest_error = error;
            best_centers = centers;
        }
    }    

    //std::cout << "Data:" << std::endl;
    //printMat(data);
    //matrix2csv(data, "temp.csv");
    //std::cout << "Centers: " << std::endl;
    //printMat(best_centers);

    return mytocv(best_centers);
}

std::vector<int> zeroton(int n){
    std::vector<int> retval;
    for(int i = 0; i < n; i++){
        retval.push_back(i);
    }
    return retval;
}

std::vector<int> randvector(int n, int K){
    std::vector<int> retval;
    for(int i = 0; i < n; i++){
        retval.push_back(rand()%K);
    }
    return retval;
}

void plusequals(std::vector<float>& v1, std::vector<float>& v2){
    for(int i = 0; i < v1.size(); i++){
        v1[i] += v2[i];
    }
}

void divequals(std::vector<float>& v, float d){
    for(auto& a : v){
        a /= d;
    }
}

cv::Mat fastkmeans2(cv::Mat input, int K, int epochs){
    mymatrix data = cvtomy(input);
    std::cout << "Data, " << data.size() << " by " << data[0].size() << std::endl;
    //printMat(data); 
   
    // assign WIDTH
    float WIDTH = .05;

    // initialize centers matrix
    mymatrix centers = zeroMatrix(K, data[0].size());
    // initialize intervals
    //std::vector<std::vector<int>> intervals((int)(2/WIDTH));
    //intervals[0] = zeroton(data.size());
    std::vector<float> intervals(data.size(), -.5); // use variable instead of intervals

    // bins left of this are negative/to be revisited
    int middle = 1/WIDTH;

    // store the closest center to each point
    // initialize as random but guarantee each center gets at least one point
    std::vector<int> closest_centers = randvector(data.size(), K);
    for(int i = 0; i < K; i++){
        closest_centers[i] = i;
    }
    
    // terminates loop
    bool done = false;

    // max movement of center
    float D = 0;
    // to terminate early
    int run = 0;

    // first time populate centers
    {
        mymatrix totals = zeroMatrix(K, data[0].size());
        std::vector<int> counts = std::vector<int>(K);

        for(int i = 0; i < data.size(); i++){
            int cc = closest_centers[i];
            plusequals(totals[cc], data[i]);
            counts[cc]++;
        }

        for(int i = 0; i < K; i++){
            if(counts[i]){
                divequals(totals[i], counts[i]);
                centers[i] = totals[i];
            }
        }
    }

    
    // sorts two doubles together based on first
    auto sortdoubles = [](std::vector<float>& v, std::vector<int>& v2) -> void{
        if(v[0] <= v[1]) return;
        float temp = v[0];
        v[0] = v[1];
        v[1] = temp;
        int temp2 = v2[0];
        v2[0] = v2[1];
        v2[1] = temp2;
    };

    while(!done && run++ < epochs){
        /*
        std::cout << "Run number " << run << std::endl;
        std::cout << "Centers: " << std::endl;
        printMat(centers);
        std::cout << "Closest centers to each point:" << std::endl;
        for(auto& cc : closest_centers){
            std::cout << cc << " ";
        }
        std::cout << std::endl << "==============================" << std::endl << std::endl;
        */

        // to calculate new centers
        mymatrix totals = zeroMatrix(K, data[0].size());
        std::vector<int> counts = std::vector<int>(K);

        // sorts two doubles together based on first
        auto sortdoubles = [](std::vector<float>& v, std::vector<int>& v2) -> void{
            if(v[0] <= v[1]) return;
            float temp = v[0];
            v[0] = v[1];
            v[1] = temp;
            int temp2 = v2[0];
            v2[0] = v2[1];
            v2[1] = temp2;
        };

        int visited = 0;

        done = true;
        // all points
        for(int i = 0; i < data.size(); i++){
            if((intervals[i] -= 2*D) < 0){ // calculate center
                visited++;

                done = false;

                // find closest and second closest centers
                auto& point = data[i];
                std::vector<float> dclosest_distances = {10, 10};
                std::vector<int> dclosest_centers = {-1, -1};

                for(int j = 0; j < K; j++){ // check against all centers
                    float d = distance(point, centers[j]);
                    if(d < dclosest_distances[1]){
                        dclosest_distances[1] = d;
                        dclosest_centers[1] = j;
                        sortdoubles(dclosest_distances, dclosest_centers);
                    }
                }

                // FIXME
                /*
                std::cout << "Run " << run << " point " << i << " dcc and dcd" << std::endl;
                for(auto& a : dclosest_centers){
                    std::cout << a << " ";
                } std::cout << std::endl;
                for(auto& a : dclosest_distances){
                    std::cout << a << " ";
                } std::cout << std::endl;
                */

                // set closest center
                closest_centers[i] = dclosest_centers[0];
                // find new interval value
                intervals[i] = dclosest_distances[1] - dclosest_distances[0];
            }

            // used to find new center positions
            plusequals(totals[closest_centers[i]], data[i]);
            ++counts[closest_centers[i]];

        }

        std::cout << "Visited " << visited << "/" << data.size() << " = " << (float)visited/data.size() * 100 << "%" << std::endl;

        // reset max distance
        D = 0;
        // move centers, set D
        for(int i = 0; i < K; i++){
            if(counts[i]){
                divequals(totals[i], counts[i]);
                auto dis = distance(centers[i], totals[i]);
                if(dis > D){
                    D = dis;
                }
                centers[i] = totals[i];
            }
        }

        // shift intervals by 2D
        //for(auto& interv : intervals){
        //    interv -= 2*D;
        //}

    }

    std::cout << "Done with " << run << " runs" << std::endl;
    /*std::cout << "Done with " << run << " runs, centers:" << std::endl;
    printMat(centers);
    std::cout << "Closest centers to each point:" << std::endl;
    for(auto& cc : closest_centers){
        std::cout << cc << " ";
    }
    std::cout << std::endl;*/

    return mytocv(centers);
}

cv::Mat kmeans2(cv::Mat input, int K, int attempts, cv::Mat centers){
    
}
