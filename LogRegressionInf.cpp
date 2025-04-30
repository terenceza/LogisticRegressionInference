
#include "LogRegressionInf.hpp"
#include <algorithm>
#include <memory>

void LogRegressionInf(  int* DataInBuff,  int* PredictBuff,
			int* WeightsBuff ,int* BiasP,
			unsigned int* DataDimensionP, unsigned int* NumSamplesP)
{
	#pragma HLS INTERFACE s_axilite port=return bundle=CONTROL_BUS

	#pragma HLS INTERFACE mode=m_axi port=DataInBuff   offset=slave bundle=gmem
	#pragma HLS INTERFACE mode=m_axi port=WeightsBuff  offset=slave bundle=gmem
	#pragma HLS INTERFACE mode=m_axi port=PredictBuff  offset=slave bundle=gmem

        #pragma HLS INTERFACE mode=s_axilite port=BiasP
        #pragma HLS INTERFACE mode=s_axilite port=DataDimensionP
	#pragma HLS INTERFACE mode=s_axilite port=NumSamplesP

	#pragma HLS INTERFACE mode=s_axilite port=DataInBuff
	#pragma HLS INTERFACE mode=s_axilite port=WeightsBuff
	#pragma HLS INTERFACE mode=s_axilite port=PredictBuff


	unsigned int DataDim    = *DataDimensionP; // number of features per sample
	unsigned int NumSamples = *NumSamplesP;    // number of samples

	//printf("%d %d %d %d %d \n", DataDimension, HiddenDimension, FullDimension, LearningRate, Momentum);


	DataType Inputs  [MAX_DATA_SIZE * MAX_TEST_SAMPLES]; 	// an array of all samples with their features
	DataType Weights [MAX_DATA_SIZE]; 						// number of features per sample - each feature associated with a weight
	DataType Predicts[MAX_TEST_SAMPLES]; 					// Prediction for each test sample
	DataType Bias;

	float fBias;
	memcpy(&fBias, BiasP, sizeof(int));
	Bias = (DataType)fBias;

	// copy DataInBuff to local array
	CopyIntToDataTypeBuffers(DataInBuff, Inputs, DataDim * NumSamples);

	CopyIntToDataTypeBuffers(WeightsBuff, Weights, DataDim);

	Predict(Inputs, Predicts, Weights, Bias, DataDim, NumSamples);

	CopyDataTypeToIntBuffers(Predicts, PredictBuff, NumSamples);
}

// Inputs: data (num_features, num_samples) - now a 1D array
// w: weights (num_features, 1) - now a 1D array
// b: bias (scalar)
// num_features: number of features (e.g., num_features * num_features * 3 for an image)
// num_samples: number of training samples

void Predict(   DataType* Inputs, DataType* Predictions,
		 	 	DataType* Weights, DataType Bias,
				unsigned int NumFeatures, unsigned int NumSamples)
{

    // Inputs: data (num_features, num_samples) - now a 1D array
    // Weights: weights (num_features, 1) - now a 1D array
    // Bias: bias (scalar)
    // NumFeatures: number of features (e.g., num_features * num_features * 3 for an image)
    // NumSamples: number of test samples

    int i, j;
    DataType z;

    //printf("%d features, %d samples \n", num_features, num_samples);

    // Prediction

    Loop1:
    for (i = 0; i < NumSamples; i++)
    {
    	//printf("sample %d \n", i);

        z = Bias;
        Loop2:
        for (j = 0; j < NumFeatures; j++)
        {
			#pragma HLS PIPELINE off
        	//printf("feature (%d,%d) Input: %f Weight: %f \n", i,j, Inputs[i * num_features + j],  w[j]);

            z += Weights[j] * Inputs[i * NumFeatures + j]; // Indexing into Inputs

        }

        Predictions[i] = sigmoid(z);
    }
}


//The sigmoid function  can assume all values in the range ]0, 1[.
DataType sigmoid(DataType d)
{
	DataType Res = (DataType)1.0 / ( (DataType)1.0 + (DataType)exp(-d));

	//printf("Res(%f) = %f", d, Res);

	return Res;
}


void CopyIntToDataTypeBuffers(int* From, DataType* To, unsigned int Dim)
{
	Loop3:
	float fval;
	for (int i = 0 ; i < Dim ; ++i)
	{
		memcpy(&fval, &From[i], sizeof(float));
		To[i] = static_cast<DataType>(fval);
	}
}

void CopyDataTypeToIntBuffers(DataType* From, int* To, unsigned int Dim)
{
	Loop3:
	float fval;
	for (int i = 0 ; i < Dim ; ++i)
	{

		fval = static_cast<float>(From[i]);
		memcpy(&To[i], &fval, sizeof(float));

	}
}

DataType Relu(DataType x)
{
    return (x > 0) ? x : (DataType)0; // Use ternary operator for ap_fixed
}
