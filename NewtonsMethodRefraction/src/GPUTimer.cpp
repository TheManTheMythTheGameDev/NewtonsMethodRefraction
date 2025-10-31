#include "GPUTimer.h"

GPUTimer::GPUTimer()
{
    glGenQueries(2, queryID);
}

GPUTimer::~GPUTimer()
{
    glDeleteQueries(2, queryID);
}

void GPUTimer::Begin()
{
    glQueryCounter(queryID[0], GL_TIMESTAMP);
}

GLint64 GPUTimer::End()
{
    glQueryCounter(queryID[1], GL_TIMESTAMP);
    GLint stopTimerAvailable = 0;

    // This stalls the CPU until the GPU is done
    while (!stopTimerAvailable)
    {
        glGetQueryObjectiv(queryID[1], GL_QUERY_RESULT_AVAILABLE, &stopTimerAvailable);
    }

    glGetQueryObjectui64v(queryID[0], GL_QUERY_RESULT, &startTime);
    glGetQueryObjectui64v(queryID[1], GL_QUERY_RESULT, &endTime);

    return endTime - startTime;
}
