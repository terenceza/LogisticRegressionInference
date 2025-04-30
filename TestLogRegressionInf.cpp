
/*
 * Sensecoding.com Integrated Solutions for Low-Powered Systems
 * C Simulation on Vitis HLS
 */


#include "../src/LogRegressionInf.hpp"
#include <stdio.h>
#include <string>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <vector>
#include <cerrno>
#include <stdlib.h>
#include <chrono>

using namespace std;

int main()
{
    unsigned int DataDim = 4;
	unsigned int TestSize = 10;

	const std::string filename("irisdata.txt");
    std::vector<int> data;

    std::ifstream file(filename);
    std::string line;

    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {}; // Return an empty vector to indicate failure
    }

    int lineCount = 0;
    while (std::getline(file, line) && lineCount < TestSize) // Read line by line, limit to 100 lines
    {
        std::stringstream ss(line);
        std::string value;
        int valueCount = 0;
        //printf("\n");
        while (std::getline(ss, value, ',') && valueCount < 5) // Read values separated by commas, limit to 4
        {
            try
            {
        		int   ival;
        		float fval = std::stof(value);
        		memcpy(&ival,&fval, sizeof(float));

            	if (valueCount < 4)
            	{
            		data.push_back(ival);
            	}

                //printf("%f ", std::stof(value));
            }
            catch (const std::invalid_argument &e)
            {
                std::cerr << "Error: Invalid number format in file " << filename << " at line " << lineCount + 1 << ", value: " << value << std::endl;
            }
            valueCount++;
        }

        lineCount++;
    }
    if (lineCount < MAX_TEST_SAMPLES)
    {
        std::cerr << "Warning: File " << filename << " has less than 100 lines. Read " << lineCount << " lines." << std::endl;
    }
    file.close();

	// push 1 weight per feature
    float fWeights[4] = { -0.562098, -0.596679, 1.066087, 0.882578 };
    int   Weights[4];

    for (int i = 0 ; i < 4 ; ++i)
    {
    	memcpy(&Weights[i], &fWeights[i], sizeof(int));
    	//printf(" w%d=0x%04x ", Weights[i]);
    }


    float fbias =  -0.087511;
    int   Bias;
    memcpy(&Bias, &fbias, sizeof(float));

	int Predictions[MAX_TEST_SAMPLES];
	float fPredictions[MAX_TEST_SAMPLES];

	auto start = std::chrono::high_resolution_clock::now();

	LogRegressionInf(data.data(), Predictions, Weights, &Bias, &DataDim, &TestSize);

	for (int i = 0 ; i < MAX_TEST_SAMPLES ; ++i)
	{
		memcpy(&fPredictions[i], &Predictions[i],  sizeof(int));
	}

	for (int i = 0 ; i < MAX_TEST_SAMPLES ; ++i)
	{
		printf(" %f ", fPredictions[i]);
	}

	auto stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start).count();
	 std::cout << "RUNTIME : "
			   << stop <<" ms " << std::endl;

	 return 0;

}
