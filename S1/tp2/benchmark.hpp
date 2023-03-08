#include <chrono>
#include <iostream>

class Benchmark {
public:
	Benchmark(std::string message);
	~Benchmark();

private:
	std::string m_message;

	decltype(std::chrono::high_resolution_clock::now()) m_start = std::chrono::high_resolution_clock::now();
};
