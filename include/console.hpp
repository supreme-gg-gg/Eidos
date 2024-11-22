#ifndef CONSOLE_H
#define CONSOLE_H

#include <string>
#include <iostream>

class Console {
public:
    // simulate enum
    
    using Flags = unsigned char;
    static const Flags INFO = 0;
    static const Flags WARNING = 1;
    static const Flags ERROR = 2;
    static const Flags DEBUG = 3;
    static const Flags WORSHIP = 4;
    
    Console(bool debugMode=false, bool treatWarningAsError=false, bool onlyLogErrors = false, bool quietMode=false) {
        this->config(debugMode, treatWarningAsError, onlyLogErrors, quietMode);
    };
    
    // Prints a message to standard output, styled with the specified message flag.
    inline void log(std::string content, Console::Flags flag = Console::INFO) {
        if (this->quietMode || (flag != Console::ERROR && this->onlyLogErrors) || (flag == Console::DEBUG && !this->debugMode)) {
            return;
        }
        if (flag == Console::WARNING && this->onlyLogErrors) {
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
        output += content + "\n";
        if (flag == Console::ERROR) {
            std::cerr << output << std::endl;
        }
        else {
            std::cout << output << std::endl;
        }
    }
    
    // Prints a message to standard output, styled with the specified message flag.
    inline void log(std::wstring content, Console::Flags flag = Console::INFO) {
        if (this->quietMode || (flag != Console::ERROR && this->onlyLogErrors) || (flag == Console::DEBUG && !this->debugMode)) {
            return;
        }
        if (flag == Console::WARNING && this->onlyLogErrors) {
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
        output += content + L"\n";
        if (flag == Console::ERROR) {
            std::wcerr << output << std::endl;
        }
        else {
            std::wcout << output << std::endl;
        }
    }
    
    bool debugMode;
    bool treatWarningAsError;
    bool onlyLogErrors;
    bool quietMode; // Suppress all messages including errors
    inline void config(bool debugMode=false, bool treatWarningAsError=false, bool onlyLogErrors = false, bool quietMode=false) {
        this->debugMode = debugMode;
        this->treatWarningAsError = treatWarningAsError;
        this->onlyLogErrors = onlyLogErrors;
        this->quietMode = quietMode;
    }
};

Console console;

#endif //CONSOLE_H