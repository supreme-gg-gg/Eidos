#ifndef UTILS_H
#define UTILS_H

#include <string>

enum msgFlags {
    INFO = 0,
    WARNING,
    ERROR,
    DEBUG,
    WORSHIP
};

// Output a message to standard output stream styled with the specified message flag.
void printMsg(std::string content, msgFlags flag);

#endif //UTILS_H