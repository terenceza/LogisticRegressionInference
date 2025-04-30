#ifndef LOG_REG_ING_HPP
#define LOG_REG_ING_HPP

#include <ap_fixed.h>

#ifdef __SYNTHESIS__
	#include <hls_math.h>
#else
	#include <cmath>
#endif

#define MAX_DATA_SIZE      4
#define MAX_TEST_SAMPLES  10
#define MAX_EPOCHS       100

#ifdef __SYNTHESIS__
	typedef ap_fixed<25, 5, AP_RND_ZERO> DataType;
#else
	typedef float DataType;
	typedef double CoeffType;
#endif

void LogRegressionInf(int* DataInBuff,  int* PredictBuff,
		      int* WeightsBuff ,int* BiasP,
		      unsigned int* DataDimensionP, unsigned int* NumSamplesP);

void Predict(   DataType* Inputs, DataType* Predictions,
		DataType* Weights, DataType Bias,
		unsigned int NumFeatures, unsigned int NumSamples);

DataType sigmoid(DataType d);
DataType Relu(DataType x);

void CopyIntToDataTypeBuffers(int* From, DataType* To, unsigned int Dim);
void CopyDataTypeToIntBuffers(DataType* From, int* To, unsigned int Dim);

#endif
