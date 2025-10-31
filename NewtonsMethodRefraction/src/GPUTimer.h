#pragma once
#include "external/glad.h"

// The GPU and CPU are not synchronized
// To time GPU operations, we must use special timers
class GPUTimer
{
public:
    // Ideally, these timers should only be initialized once when the application is launched
    GPUTimer();
    ~GPUTimer();

    // The Begin() operation is fast
    void Begin();
    // The End() operation forces the CPU and GPU to synchronize and so is very slow
    // Returns the number of nanoseconds that the GPU operations have taken since Begin() was called
    GLint64 End();
private:
    GLuint64 startTime, endTime;
    unsigned int queryID[2];
};
