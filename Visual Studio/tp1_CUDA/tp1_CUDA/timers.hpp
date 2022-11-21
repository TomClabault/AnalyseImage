#include <cuda_runtime.h>

#include <cstdlib>
#include <chrono>
#include <iostream>

class Timer
{
public:
    Timer(std::string message, int iterations);
    ~Timer();

private:
    std::string m_message;

    unsigned int m_iterations;

    std::chrono::time_point<std::chrono::steady_clock> m_start, m_stop;
};

class CudaTimer
{
public:
    CudaTimer(std::string message, int iterations);
    ~CudaTimer();

private:
    std::string m_message;

    unsigned int m_iterations;

    cudaEvent_t m_start, m_stop;
};