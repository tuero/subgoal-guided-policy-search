// File: init.h
// Description: Initialize torch reproducibility and logging

#ifndef HPTS_COMMON_INIT_H_
#define HPTS_COMMON_INIT_H_

#include <cstdint>
#include <string>

namespace hpts {

/**
 * Initialize torch reproducibility
 * @param seed The seed to initialize torch rngs
 */
void init_torch(uint64_t seed);

/**
 * Initialize the file and terminal loggers
 * @param console_only Flag to only log to console
 * @param path The directory which the experiment output resides
 * @param postfix Postfix for logger name file
 * @param erase_if_exists Erase if log file already exists
 */
void init_loggers(
    bool console_only = true,
    const std::string &path = "",
    const std::string &postfix = "",
    bool erase_if_exists = true
);

/**
 * Log the invoked command used to run the current program
 * @param argc Number of arguments
 * @param argv char array of params
 */
void log_flags(int argc, char **argv);

/**
 * Flush the logs
 */
void log_flush();

/**
 * Close all loggers
 */
void close_loggers();

}    // namespace hpts

#endif    // HPTS_COMMON_INIT_H_
