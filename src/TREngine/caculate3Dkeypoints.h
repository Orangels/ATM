#pragma once
#pragma 

void caculate3Dkeypoints(float* vikeyPoints, float* vokeyPoints, int vBatchSize, float* vpBoxes, cudaTextureObject_t vShpBase, cudaTextureObject_t vExpBase, cudaStream_t vCudaStream);