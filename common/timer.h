#ifndef TIMER_H
#define TIMER_H

#include <string>
#include <memory>

struct TimerData;

class Timer
{
public:
    Timer(const std::string& message = "timer");
    ~Timer();

    void setMessage(const std::string& message = "timer") { this->message = message; }

private:
    std::string message;
    std::shared_ptr<TimerData> data;
};

#endif // TIMER_H
