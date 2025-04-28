
#include "LogRegressionInf.hpp"
#include <algorithm>
#include <memory>

void LogRegressionInf(int* DataInBuff,  int* PredictBuff,
					  int* WeightsBuff)
					//, int* BiasP,
				    //  unsigned int* DataDimensionP, unsigned int* NumSamplesP)
{
	#pragma HLS INTERFACE s_axilite port=return bundle=CONTROL_BUS

	#pragma HLS INTERFACE mode=m_axi port=DataInBuff   depth=256 offset=slave bundle=gmem
	#pragma HLS INTERFACE mode=m_axi port=WeightsBuff  depth=256 offset=slave bundle=gmem
	#pragma HLS INTERFACE mode=m_axi port=PredictBuff  depth=256 offset=slave bundle=gmem

//    #pragma HLS INTERFACE mode=s_axilite port=BiasP
//    #pragma HLS INTERFACE mode=s_axilite port=DataDimensionP
//	 #pragma HLS INTERFACE mode=s_axilite port=NumSamplesP

	#pragma HLS INTERFACE mode=s_axilite port=DataInBuff
	#pragma HLS INTERFACE mode=s_axilite port=WeightsBuff
	#pragma HLS INTERFACE mode=s_axilite port=PredictBuff


	unsigned int DataDim    = 4; //*DataDimensionP; // number of features per sample
	unsigned int NumSamples = 10; //*NumSamplesP;    // number of samples

	//printf("%d %d %d %d %d \n", DataDimension, HiddenDimension, FullDimension, LearningRate, Momentum);


	DataType Inputs  [MAX_DATA_SIZE * MAX_TEST_SAMPLES]; 	// an array of all samples with their features
	DataType Weights [MAX_DATA_SIZE]; 						// number of features per sample - each feature associated with a weight
	DataType Predicts[MAX_TEST_SAMPLES]; 					// Prediction for each test sample

	DataType Bias = static_cast<DataType>(-0.087511);

//	#pragma HLS ARRAY_PARTITION variable=Inputs   type=complete
//	#pragma HLS ARRAY_PARTITION variable=Weights  type=complete
//	#pragma HLS ARRAY_PARTITION variable=Predicts type=complete




	// copy DataInBuff to local array
	CopyFloatToDataTypeBuffers(DataInBuff, Inputs, DataDim * NumSamples);

	CopyFloatToDataTypeBuffers(WeightsBuff, Weights, DataDim);

	Predict(Inputs, Predicts, Weights, Bias, DataDim, NumSamples);

	CopyDataTypeToFloatBuffers(Predicts, PredictBuff, NumSamples);

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
    // Label: labels (1, num_samples)  - now a 1D array
    // w: weights (num_features, 1) - now a 1D array
    // b: bias (scalar)
    // num_features: number of features (e.g., num_features * num_features * 3 for an image)
    // num_samples: number of training samples
    // dw: gradient of cost with respect to w (output, same shape as w) - 1D array
    // db: gradient of cost with respect to b (output, scalar)
    // cost: cost of the logistic regression (output, scalar)

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

//        printf("\n");
//        printf("fval=%f ", Predictions[i]);
//        printf("feature (%d,%d) Input: %f  z: %f sigmoid: %f \n", i,j, Inputs[i * NumFeatures + j],  z, Predictions[i]);
    }

}


//The sigmoid function  can assume all values in the range ]0, 1[.
DataType sigmoid(DataType d)
{
	DataType Res = (DataType)1.0 / ( (DataType)1.0 + (DataType)exp(-d));

	//printf("Res(%f) = %f", d, Res);

	return Res;
}


void CopyFloatToDataTypeBuffers(int* From, DataType* To, unsigned int Dim)
{
	Loop3:
	float fval;
	for (int i = 0 ; i < Dim ; ++i)
	{
		memcpy(&fval, &From[i], sizeof(float));
		To[i] = static_cast<DataType>(fval);
	}
}

void CopyDataTypeToFloatBuffers(DataType* From, int* To, unsigned int Dim)
{
	Loop3:
	float fval;
	for (int i = 0 ; i < Dim ; ++i)
	{

		fval = static_cast<float>(From[i]);
		//printf("fval=%f ", fval);
		memcpy(&To[i], &fval, sizeof(float));

	}
}

DataType Relu(DataType x)
{
    return (x > 0) ? x : (DataType)0; // Use ternary operator for ap_fixed
}
