/******************************************************************************
* Sensecoding.com free license
* Run on Vitis 
* Weights and Bias were generated with LogRegressionLearning 
******************************************************************************/

#ifndef LOG_REG__H
#define LOG_REG__H

#define MAX_LINE_LENGTH 100
#define DDR_DATA_BASE_ADDR_IN       0x0000000000000101800
#define DDR_PREDICT_BASE_ADDR       0x0000000000000101C00
#define DDR_WEIGHTS_BASE_ADDR       0x0000000000000101D00

#define NUM_SAMPLES 10
#define NUM_FEATURES 4

float testData[] = 
{
    6.7, 3.1, 5.6, 2.4,
    6.9, 3.1, 5.1, 2.3,
    5.8, 2.7, 5.1, 1.9,
    6.8, 3.2, 5.9, 2.3,
    6.7, 3.3, 5.7, 2.5,
    6.7, 3.0, 5.2, 2.3,
    6.3, 2.5, 5.0, 1.9,
    6.5, 3.0, 5.2, 2.0,
    6.2, 3.4, 5.4, 2.3,
    5.9, 3.0, 5.1, 1.8
};

float Weights[] = 
{
    -0.562098, -0.596679, 1.066087, 0.882578
};

float Predictions[10] = { 1.1, 1.2, 1.3, 1.4, 1.5, 1.1, 1.2, 1.3, 1.4, 1.5};

float Bias = -0.087511;

#endif
