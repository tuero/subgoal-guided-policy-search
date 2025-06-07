// File: stop_token.h
// Description: std::stop_token like flag class to signal for threads

#ifndef HPTS_UTIL_STOP_TOKEN_H_
#define HPTS_UTIL_STOP_TOKEN_H_

#include <atomic>

namespace hpts {

class StopToken {
public:
    void stop() noexcept {
        flag_ = true;
    }

    [[nodiscard]] auto stop_requested() const noexcept -> bool {
        return flag_;
    }

private:
    std::atomic<bool> flag_{false};
};

}    // namespace hpts

#endif    // HPTS_UTIL_STOP_TOKEN_H_
