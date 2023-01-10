#include "benchmark.hpp"

Benchmark::Benchmark(std::string message) {
	m_start = std::chrono::high_resolution_clock::now();

	m_message = message;
}

Benchmark::~Benchmark() {
	m_stop = std::chrono::high_resolution_clock::now();

	std::cout << m_message << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(m_stop - m_start).count() << "ms" << std::endl;
}