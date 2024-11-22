#ifndef CONSOLE_H
#define CONSOLE_H

#include <string>
#include <iostream>

namespace Console {
    enum Flags {
        INFO = 0,
        WARNING,
        ERROR,
        DEBUG,
        WORSHIP
    };

    extern bool debugMode;
    extern bool treatWarningAsError;
    extern bool onlyLogErrors;
    extern bool quietMode; // Suppress all messages including errors
    inline void config(bool debugMode=false, bool treatWarningAsError=false, bool onlyLogErrors = false, bool quietMode=false) {
        Console::debugMode = debugMode;
        Console::treatWarningAsError = treatWarningAsError;
        Console::onlyLogErrors = onlyLogErrors;
        Console::quietMode = quietMode;
    }

    // Prints a message to standard output, styled with the specified message flag.
    inline void log(std::string content, Console::Flags flag = Console::INFO) {
        if (Console::quietMode || (flag != Console::ERROR && Console::onlyLogErrors) || (flag == Console::DEBUG && !Console::debugMode)) {
            return;
        }
        if (flag == Console::WARNING && Console::onlyLogErrors) {
            flag = Console::ERROR;
        }
        std::string output;
        switch(flag) {
            case Console::INFO:
                output = "";
                break;
            case Console::WARNING:
                output = "[WARNING]: ";
                break;
            case Console::ERROR:
                output = "[ERROR]: ";
                break;
            case Console::DEBUG:
                output = "[DEBUG]: ";
                break;
            default:
                output = "[+69420 GEORGIST CREDIT]: ";
                break;
        }
        output += content;
        if (flag == Console::ERROR) {
            std::cerr << output << std::endl;
        }
        else {
            std::cout << output << std::endl;
        }
    }
    
    // Prints a message to standard output, styled with the specified message flag.
    inline void log(std::wstring content, Console::Flags flag = Console::INFO) {
        if (Console::quietMode || (flag != Console::ERROR && Console::onlyLogErrors) || (flag == Console::DEBUG && !Console::debugMode)) {
            return;
        }
        if (flag == Console::WARNING && Console::treatWarningAsError) {
            flag = Console::ERROR;
        }
        std::wstring output;
        switch(flag) {
            case Console::INFO:
                output = L"";
                break;
            case Console::WARNING:
                output = L"[WARNING]: ";
                break;
            case Console::ERROR:
                output = L"[ERROR]: ";
                break;
            case Console::DEBUG:
                output = L"[DEBUG]: ";
                break;
            default:
                output = L"[+69420 GEORGIST CREDIT]: ";
                break;
        }
        output += content;
        if (flag == Console::ERROR) {
            std::wcerr << output << std::endl;
        }
        else {
            std::wcout << output << std::endl;
        }
    }    
}

#endif //CONSOLE_H