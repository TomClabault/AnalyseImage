#include "timers.hpp"

Timer::Timer(std::string message, int iterations)
{
    m_message = message;
    m_iterations = iterations;
    m_start = std::chrono::high_resolution_clock::now();
}

Timer::~Timer()
{
    m_stop = std::chrono::high_resolution_clock::now();

    long long int duration = std::chrono::duration_cast<std::chrono::milliseconds>(m_stop - m_start).count();

    std::cout << "[CPU] " << m_message << ": " << duration << "ms total | " << (duration / (double)m_iterations) << "ms average" << std::endl;
}

CudaTimer::CudaTimer(std::string message, int iterations) : m_message(message), m_iterations(iterations)
{
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);

    cudaEventRecord(m_start);
}

CudaTimer::~CudaTimer()
{
    cudaEventRecord(m_stop);
    cudaEventSynchronize(m_stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, m_start, m_stop);

    std::cout << "[GPU] " << m_message << ": " << milliseconds << "ms total | " << (milliseconds / (double)m_iterations) << "ms average" << std::endl;
}
