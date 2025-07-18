#pragma once // Or use traditional include guards

#include <cstdio>        // For snprintf, fprintf, stderr
#include <string>        // For std::string (used in std::runtime_error, std::invalid_argument)
#include <stdexcept>     // For std::runtime_error, std::invalid_argument

#ifndef THROW_IF_ERROR_CONDITION
#define THROW_IF_ERROR_CONDITION(condition, ExceptionType, format_str, ...) \
    do {                                                                    \
        if (condition) {                                                    \
            char errorMsgBuffer[2048]; /* Buffer for the formatted message */ \
            /* Format the error message using snprintf */                   \
            snprintf(errorMsgBuffer, sizeof(errorMsgBuffer), format_str, ##__VA_ARGS__); \
            /* Print to stderr (optional, but good for logging) */          \
            fprintf(stderr, "Error in %s (line %d): %s\n", __FILE__, __LINE__, errorMsgBuffer); \
            /* Throw the specified exception type */                        \
            throw ExceptionType(errorMsgBuffer);                            \
        }                                                                   \
    } while (0)
#endif