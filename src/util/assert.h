// File: assert.h
// Description: Custom assertion for release builds

#ifndef HPTS_UTIL_ASSERT_H_
#define HPTS_UTIL_ASSERT_H_

#include <exception>
#include <iostream>
#include <sstream>
#include <string>

namespace hpts {

class HPTSException : public std::exception {
public:
    HPTSException(
        const std::string &input_message,
        const char *file_name,
        const char *function_signature,
        int line_number
    ) {
        std::ostringstream os;
        os << "File: " << file_name << ":" << line_number << std::endl;
        os << "In function: " << std::endl;
        os << "\t" << function_signature << std::endl;
        os << std::endl;
        os << "HPTS Error: " << input_message << std::endl;
        message = os.str();
    }
    [[nodiscard]] auto what() const noexcept -> const char * override {
        return message.c_str();
    }

private:
    std::string message;
};

// NOLINTNEXTLINE(*-macro-usage,*-pro-bounds-array-to-pointer-decay)
#define HPTS_EXCEPTION(msg) throw HPTSException(msg, __FILE__, __PRETTY_FUNCTION__, __LINE__)

// NOLINTNEXTLINE(*-macro-usage,*-pro-bounds-array-to-pointer-decay)
#define HPTS_ERROR(msg)                                                        \
    {                                                                          \
        try {                                                                  \
            throw HPTSException(msg, __FILE__, __PRETTY_FUNCTION__, __LINE__); \
        } catch (const TAException &e) {                                       \
            std::cout << e.what() << std::endl;                                \
        }                                                                      \
        std::exit(1);                                                          \
    }

// NOLINTNEXTLINE(*-macro-usage,*-pro-bounds-array-to-pointer-decay)
#define HPTS_ASSERT(cond, msg) if (!cond)

template <typename T>
inline void hpts_assert(const T &assertion) {
    if (!assertion) {
        HPTS_EXCEPTION("");
    }
}
}    // namespace hpts

#endif    // HPTS_UTIL_ASSERT_H_
