#ifndef CRYSTAL_NET_H
#define CRYSTAL_NET_H

#define INPUT_DIM 784
#define OUTPUT_DIM 10
#define NUM_NEURONS 32

void crystal_forward(const float* input, float* output);
int crystal_predict(const float* input);

#endif
