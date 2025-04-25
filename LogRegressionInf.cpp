
#include "LogRegressionInf.hpp"
#include <algorithm>
#include <memory>

void LogRegressionInf(DataType* DataInP,  DataType* PredicP,
		 	 	      DataType* WeightsP, DataType* BiasP,
				      unsigned int* DataDimensionP, unsigned int* NumSamplesP,
				      DataType* dw, DataType* db, DataType* cost)
{
	#pragma HLS INTERFACE s_axilite port=return bundle=CONTROL_BUS

	#pragma HLS INTERFACE mode=m_axi port=DataInP    offset=slave bundle=gmem
	#pragma HLS INTERFACE mode=m_axi port=WeightsP   offset=slave bundle=gmem
	#pragma HLS INTERFACE mode=m_axi port=PredicP    offset=slave bundle=gmem

    #pragma HLS INTERFACE mode=s_axilite port=BiasP
    #pragma HLS INTERFACE mode=s_axilite port=DataDimensionP
	#pragma HLS INTERFACE mode=s_axilite port=NumSamplesP
    #pragma HLS INTERFACE mode=s_axilite port=dw
    #pragma HLS INTERFACE mode=s_axilite port=db
	#pragma HLS INTERFACE mode=s_axilite port=cost

	#pragma HLS INTERFACE mode=s_axilite port=DataInP

	unsigned int DataDim    = *DataDimensionP; // number of features per sample
	unsigned int NumSamples = *NumSamplesP;    // number of samples

	//printf("%d %d %d %d %d \n", DataDimension, HiddenDimension, FullDimension, LearningRate, Momentum);


	DataType Inputs  [MAX_DATA_SIZE * MAX_TEST_SAMPLES]; 	// an array of all samples with their features
	DataType Weights [MAX_DATA_SIZE]; 						// number of features per sample - each feature associated with a weight
	DataType Predicts[MAX_TEST_SAMPLES]; 					// Prediction for each test sample

	#pragma HLS ARRAY_PARTITION variable=Inputs   type=complete
	#pragma HLS ARRAY_PARTITION variable=Weights  type=complete
	#pragma HLS ARRAY_PARTITION variable=Predicts type=complete


	// copy inputs to local array
	//printf("\n Inputs : \n");
	for (int i = 0 ; i < DataDim * NumSamples; ++i)
	{
		Inputs[i] = DataInP[i];
		//printf("%f ", Inputs[i]);
	}


	//printf("\n Weights : \n");

	for (int w = 0 ; w < DataDim; ++w)
	{
		Weights[w] = WeightsP[w];
		//printf("%f ", Weights[w]);

	}

	Predict(Inputs, Predicts, Weights, *BiasP, DataDim, NumSamples);
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
    DataType A[MAX_TEST_SAMPLES]; // Allocate up to 150 samples

    for (i = 0; i < NumSamples; i++)
    {
    	//printf("sample %d \n", i);
		#pragma HLS PIPELINE off

        z = Bias;
        for (j = 0; j < NumFeatures; j++)
        {
			#pragma HLS PIPELINE off
        	//printf("feature (%d,%d) Input: %f Weight: %f \n", i,j, Inputs[i * num_features + j],  w[j]);

            z += Weights[j] * Inputs[i * NumFeatures + j]; // Indexing into Inputs

        }

        Predictions[i] = sigmoid(z);
        //printf("feature (%d,%d) Input: %f  z: %f sigmoid: %f \n", i,j, Inputs[i * num_features + j],  z, A[i]);
    }

}


//The sigmoid function  can assume all values in the range ]0, 1[.
DataType sigmoid(DataType d)
{
	DataType Res = (DataType)1.0 / ( (DataType)1.0 + (DataType)exp(-d));

	//printf("Res(%f) = %f", d, Res);

	return Res;
}


