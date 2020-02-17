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

cv::Mat kmeans2(cv::Mat input, int K, int attempts, cv::Mat centers){
    
}
