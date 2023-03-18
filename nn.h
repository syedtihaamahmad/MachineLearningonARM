#ifndef __NN_H__
#define __NN_H__

#include <stdint.h>
#include <stdio.h>
#include "arm_math.h"
#include "arm_nnexamples_cifar10_parameter.h"
#include "arm_nnexamples_cifar10_weights.h"
#include "arm_nnfunctions.h"

void run_nn(q7_t* input_data, q7_t* output_data);


#endif
