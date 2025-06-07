// File: signaller.cpp
// Description: Signal handler + install for graceful exiting

#include "common/signaller.h"

#include <csignal>

namespace hpts {

struct SignalHandler {
    static void signal_handler([[maybe_unused]] int s) {
        if (stop_token->stop_requested()) {
            exit(1);
        } else {
            stop_token->stop();
        }
    }

    inline static auto stop_token = std::make_shared<StopToken>();    // NOLINT(*-avoid-non-const-global-variables)
};

/**
 * Create and install a signal handler
 * On SIGINT, token will request stop, and all objects storing it will call their exit code
 */
std::shared_ptr<StopToken> signal_installer() {
    std::signal(SIGINT, &SignalHandler::signal_handler);
    return SignalHandler::stop_token;
}

/**
 * Create and install a signal handler
 * On SIGINT, token will request stop, and all objects storing it will call their exit code
 */
void signal_installer(std::shared_ptr<StopToken> stop_token) {
    SignalHandler::stop_token = stop_token;
    std::signal(SIGINT, &SignalHandler::signal_handler);
}

}    // namespace hpts
