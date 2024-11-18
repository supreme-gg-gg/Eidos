#include "../include/utils.h"
#include <iostream>
#include <string>

void printMsg(std::string content, msgFlags flag) {
    std::string output;
    switch(flag) {
        case msgFlags::INFO:
            output = "";
            break;
        case msgFlags::WARNING:
            output = "[WARNING]: ";
            break;
        case msgFlags::ERROR:
            output = "[ERROR]: ";
            break;
        case msgFlags::DEBUG:
            output = "[DEBUG]: ";
            break;
        default:
            output = "[+69420 GEORGIST CREDIT]: ";
            break;
    }
    output += content + "\n";
    std::cout << output << std::endl;
}