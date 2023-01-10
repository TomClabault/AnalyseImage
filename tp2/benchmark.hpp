#include <chrono>
#include <iostream>

class Benchmark {
public:
	Benchmark(std::string message);
	~Benchmark();

private:
	std::string m_message;

	std::chrono::steady_clock::time_point m_start, m_stop;
};
