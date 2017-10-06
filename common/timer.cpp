#include "timer.h"
#include <chrono>
#include <iostream>

struct TimerData
{
    TimerData()
    {
        t_construct = std::chrono::system_clock::now();
    }
    std::chrono::system_clock::time_point t_construct;
};

Timer::Timer(const std::string& message) : message(message)
{
    data = std::make_shared<TimerData>();
}

Timer::~Timer()
{
    auto t_destruct = std::chrono::system_clock::now();
    std::cout << "[" << message << " : \t" << std::chrono::duration_cast<std::chrono::milliseconds>(t_destruct - data->t_construct).count() << " ms]" << std::endl;
}
