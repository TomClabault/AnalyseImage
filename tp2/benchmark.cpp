#include "benchmark.hpp"

Benchmark::Benchmark(std::string message) {
	m_message = message;
}

Benchmark::~Benchmark() {
	std::cout << m_message << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - m_start).count() << "ms" << std::endl;
}
