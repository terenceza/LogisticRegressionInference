/******************************************************************************
* Copyright (C) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/
/*
 * helloworld.c: simple test application
 *
 * Vitis app Runs on the ARM processor to submit data to FPGA for inference
*/
#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xlogregressioninf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "LogRegression.h"
#include "xscutimer.h"
#include <time.h>

#define MAX_LINE_LENGTH 100
#define DDR_DATA_BASE_ADDR_IN       0x0000000000000000
#define DDR_PREDICT_BASE_ADDR       DDR_DATA_BASE_ADDR_IN + 256
#define DDR_WEIGHTS_BASE_ADDR       DDR_DATA_BASE_ADDR_IN + 1024

XLogregressioninf LogRegInst;
XLogregressioninf_Config Cfg;

void LogRegressionInfInit(XLogregressioninf* Inst, XLogregressioninf_Config* Cfg)
//,                        u64* input_addr, u64* output_addr)
{
	if ( XLogregressioninf_Initialize(Inst, XPAR_XLOGREGRESSIONINF_0_DEVICE_ID) != XST_SUCCESS)
	{
		xil_printf("XLogregressioninf_Initialize failed \n");
	}
    else
    {
	    xil_printf("XLogregressioninf_Initialize success \n");
    }

    //XAccelconvoluteaxi_EnableAutoRestart(Inst);
	XLogregressioninf_InterruptGlobalDisable(Inst);
	XLogregressioninf_InterruptDisable(Inst, 1);
}

void WriteDataToBus()
{
    int numElements = sizeof(testData) / sizeof(testData[0]);

    xil_printf("Writing float data to FPGA registers, numelements = %d\n", numElements);

    //for (int i = 0; i < numElements;)
    int i = 0; // data index in data table
    int k = 0; // data index in memory
    int j = 0; // label index in memory

    for (i = 0 ; i < numElements; ++i)
    {
        uint32_t data32 = 0;
        memcpy(&data32, &testData[i], sizeof(float));

        XLogregressioninf_WriteReg(DDR_DATA_BASE_ADDR_IN, i*4, data32);
    }

    // for (i = 0 ; i < numElements; ++i)
    // {
    //     uint32_t data32 = 0;
    //     float    floatValue;

    //     data32 = XLogregressioninf_ReadReg(DDR_DATA_BASE_ADDR_IN, i*4);
    //     memcpy(&floatValue, &data32, sizeof(float));
    //     printf("data32: %f\n", floatValue);

    // }

    xil_printf("Finished writing data.\n");
}

void WriteWeightsToBus()
{
    int NumFeatures = 4;
    int i = 0; // data index in data table

    while (i < NumFeatures)
    {
        uint32_t data32 = 0;
        memcpy(&data32, &Weights[i], sizeof(float));

        XLogregressioninf_WriteReg(DDR_WEIGHTS_BASE_ADDR, i*4, data32);
        ++i;
    }

    // for (int i = 0 ; i < NumFeatures ; ++i)
    // {
    //     uint32_t data32 = 0;
    //     float    floatValue;

    //     data32 = XLogregressioninf_ReadReg(DDR_WEIGHTS_BASE_ADDR, i*4);
    //     memcpy(&floatValue, &data32, sizeof(float));
    //     printf("weights32: %f\n", floatValue);

    // }

    xil_printf("Finished writing weights.\n");

}



int main()
{
//    unsigned int  LoopCount = 0;
//    unsigned long TickStart = 0;
//    unsigned long TickStop = 0;
//    unsigned long Delta = 0;
//    XScuTimer	Timer;
    
    init_platform();

    LogRegressionInfInit(&LogRegInst, &Cfg);

//#if 0
    // Set start address for data
    XLogregressioninf_Set_DataInBuff(&LogRegInst, DDR_DATA_BASE_ADDR_IN);

    // Set start address for weights
    XLogregressioninf_Set_WeightsBuff(&LogRegInst, DDR_WEIGHTS_BASE_ADDR);

    //Set start address for prdictions
    XLogregressioninf_Set_PredictBuff(&LogRegInst, DDR_PREDICT_BASE_ADDR);

    WriteDataToBus();
    WriteWeightsToBus();

    u32 iBias;
//    memcpy(&iBias, &Bias, sizeof(float));
//    XLogregressioninf_Set_BiasP(&LogRegInst, iBias );
//    XLogregressioninf_Set_DataDimensionP(&LogRegInst,4);
//    XLogregressioninf_Set_NumSamplesP(&LogRegInst,10);
    //XLogregressioninf_EnableAutoRestart(&LogRegInst);

    XLogregressioninf_Start(&LogRegInst);

    while(!XLogregressioninf_IsDone(&LogRegInst)){}
    //sleep(2);
    printf("Got response !!!!!!!!!!!!\n");

    // XScuTimer_Start(&Timer);
    // TickStart = XScuTimer_GetCounterValue(&Timer);

    //while(!XLogregressioninf_IsDone(&LogRegInst))
    // while(++LoopCount < 1000000)
    // {
    //     // TickStop = XScuTimer_GetCounterValue(&Timer);
    //     // Delta = TickStop - TickStart;
    //     // if (((double)(Delta)/XPAR_PS7_CORTEXA9_0_CPU_CLK_FREQ_HZ) > 1)
    //     // {
    //     //     break;
    //     // }
    // }

    //sleep(2);

    //uint32_t bias = XLogregressioninf_Get_PredictionsBuff(&LogRegInst);
    //uint32_t cost = XLogregressioninf_Get_costP(&LogRegInst);

    for (int i = 0 ; i < 10 ; ++i)
    {
        uint32_t data32 = XLogregressioninf_ReadReg(DDR_PREDICT_BASE_ADDR, i*4);
        float floatValue;
        memcpy(&floatValue, &data32, sizeof(float));
        xil_printf("p%d = 0x%04x \n", i, data32);
    }

    // for (int i = 0 ; i < 4 ; ++i)
    // {
    //     uint32_t data32 = XLogregressioninf_ReadReg(DDR_WEIGHTS_BASE_ADDR, i*4);
    //     float floatValue;
    //     memcpy(&floatValue, &data32, sizeof(float));
    //     printf("w%d = %f \n", i, floatValue);
    // }

//    float fcost = (float)cost;
//
//    printf("Cost = %f \n", fcost);

//#endif

    
    print("Successfully ran inference application\n");
    cleanup_platform();
    return 0;
}
