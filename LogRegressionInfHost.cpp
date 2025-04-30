/******************************************************************************
* Sensecoding.com free license
* Run on Vitis 
******************************************************************************/

#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xlogregressioninf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "LogRegressionInfHost.hpp"
#include "xscutimer.h"
#include <time.h>

XLogregressioninf LogRegInst;
XLogregressioninf_Config Cfg;

void LogRegressionInfInit(XLogregressioninf* Inst, XLogregressioninf_Config* Cfg)
{
	if ( XLogregressioninf_Initialize(Inst, XPAR_XLOGREGRESSIONINF_0_DEVICE_ID) != XST_SUCCESS)
	{
		xil_printf("XLogregressioninf_Initialize failed \n");
	}
    else
    {
	    xil_printf("XLogregressioninf_Initialize success \n");
    }

	XLogregressioninf_InterruptGlobalDisable(Inst);
	XLogregressioninf_InterruptDisable(Inst, 1);
}

void NormalizeInputs()
{
    int numElements = sizeof(testData) / sizeof(testData[0]);
}

void WriteDataToBus()
{
    int numElements = sizeof(testData) / sizeof(testData[0]);

    xil_printf("Writing float input data to DDR, numElements = %d\n", numElements);

    int32_t* ddr_ptr = (int32_t*)DDR_DATA_BASE_ADDR_IN; 

    // Write the data
    for (int i = 0; i < numElements; ++i)
    {
        int32_t ival;
        memcpy(&ival, &testData[i], sizeof(int32_t));
        XLogregressioninf_WriteReg(DDR_DATA_BASE_ADDR_IN, i*4, ival);

        printf( " 0x%x ", ival);
    }

    xil_printf("Finished writing data.\n");

    // Read back and verify
    xil_printf("Reading back data from DDR for verification:\n");

    for (int i = 0; i < numElements; ++i)
    {
        int32_t val = XLogregressioninf_ReadReg(DDR_DATA_BASE_ADDR_IN, i*4);
        xil_printf("Read: 0x%x\n",  val);
    }

}

void WriteWeightsToBus()
{
    int numElements = sizeof(Weights) / sizeof(Weights[0]);

    xil_printf("Writing float weights data to DDR, numElements = %d\n", numElements);

    int32_t* ddr_ptr = (int32_t*)DDR_WEIGHTS_BASE_ADDR; 

    // Write the data
    for (int i = 0; i < numElements; ++i)
    {
        
        int32_t ival;
        memcpy(&ival, &Weights[i], sizeof(int32_t));
        XLogregressioninf_WriteReg(DDR_WEIGHTS_BASE_ADDR, i*4, ival);
        printf( " 0x%x ", ival);
    }

    xil_printf("Finished writing weights.\n");

    // Read back and verify
    xil_printf("Reading back weights from DDR for verification:\n");

    for (int i = 0; i < numElements; ++i)
    {
        float fval;
        int32_t val = XLogregressioninf_ReadReg(DDR_WEIGHTS_BASE_ADDR, i*4);
        xil_printf("Read: 0x%x \n",  val);
    }

}


int main()
{
   unsigned int  LoopCount = 0;
   unsigned long TickStart = 0;
   unsigned long TickStop = 0;
   unsigned long Delta = 0;
   XScuTimer	Timer;

   int numTestData = sizeof(testData) / sizeof(testData[0]);

    init_platform();

    LogRegressionInfInit(&LogRegInst, &Cfg);

    // Set start address for data
    XLogregressioninf_Set_DataInBuff(&LogRegInst, DDR_DATA_BASE_ADDR_IN);

    // Set start address for weights
    XLogregressioninf_Set_WeightsBuff(&LogRegInst, DDR_WEIGHTS_BASE_ADDR);

    //Set start address for predictions
    XLogregressioninf_Set_PredictBuff(&LogRegInst, DDR_PREDICT_BASE_ADDR);

    WriteDataToBus();
    WriteWeightsToBus();

    int32_t iBias;
    memcpy(&iBias, &Bias, sizeof(float));
    XLogregressioninf_Set_BiasP(&LogRegInst, iBias);
    xil_printf("Bias = 0x%x \n", iBias);

    XLogregressioninf_Set_DataDimensionP(&LogRegInst,NUM_FEATURES);
    XLogregressioninf_Set_NumSamplesP(&LogRegInst,NUM_SAMPLES);

    XLogregressioninf_Start(&LogRegInst);

    while(!XLogregressioninf_IsDone(&LogRegInst)){}
    printf("Got response !!!!!!!!!!!!\n");
    
    // XScuTimer_Start(&Timer);
    // TickStart = XScuTimer_GetCounterValue(&Timer);

    xil_printf("Read Predictions: ");
    int32_t ival;
    for (int i = 0; i < NUM_SAMPLES; ++i)
    {
        int32_t val = XLogregressioninf_ReadReg(DDR_PREDICT_BASE_ADDR, i*4);

        xil_printf(" 0x%x ",  val);
    }

    print("Successfully completed logistic inference application\n");
    cleanup_platform();
    return 0;
}
