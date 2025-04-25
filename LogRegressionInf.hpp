#ifndef AUTOENCODER_HPP
#define AUTOENCODER_HPP

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
	typedef ap_fixed<25, 5, AP_RND_ZERO> DataType, CoeffType;
#else
	typedef float DataType;
	typedef double CoeffType;
#endif

void LogRegressionInf(DataType* DataInP, DataType* PredicP,
					  DataType* WeightsP, DataType* BiasP,
					  unsigned int* DataDimensionP, unsigned int* NumSamplesP,
					  DataType* dw, DataType* db, DataType* cost);

void Predict(   DataType* Inputs, DataType* Predictions,
		 	 	DataType* Weights, DataType Bias,
				unsigned int NumFeatures, unsigned int NumSamples);

DataType sigmoid(DataType d);
DataType sigmoidDerivation(DataType d);
DataType squareError(DataType d1, DataType d2) ;
//DataType Relu(DataType d);
//DataType ReluDerivation(DataType d);

//void backpropagate();

#endif
