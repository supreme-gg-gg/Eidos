#include "../include/Eidos/csvparser.h"
#include "../include/Eidos/console.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>

std::vector<std::string> CSVParser::parseLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool inQuotes = false;
    
    for (size_t i = 0; i < line.length(); ++i) {
        char c = line[i];
        
        if (c == '"') {
            if (i + 1 < line.length() && line[i + 1] == '"') {
                // Handle escaped quotes
                field += '"';
                ++i;
            } else {
                inQuotes = !inQuotes;
            }
        } else if (c == delimiter && !inQuotes) {
            // End of field
            fields.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    
    // Check for unterminated quotes
    if (inQuotes) {
        throw std::runtime_error("Unterminated field. Expected \" before end of line.\n| "+line);
    }
    
    // Add the last field
    fields.push_back(field);
    return fields;
}

CSVParser::CSVParser(char delim /*= ','*/)
    : delimiter(delim) {}

std::vector<std::vector<std::string>> CSVParser::parse(const std::string& csvFilename) {
    filename = csvFilename;
    std::vector<std::vector<std::string>> result;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        Console::log("Could not open file \""+filename+"\".",Console::ERROR);
        return result;
    }
    
    std::string line;
    int lineNumber = 0;
    
    try {
        while (std::getline(file, line)) {
            ++lineNumber;
            if (line.empty()) continue;
            
            std::vector<std::string> fields = parseLine(line);
            result.push_back(fields);
        }
    } catch (const std::exception& e) {
        file.close();
        Console::log("In file: \""+filename+"\": line "+std::to_string(lineNumber)+": "+e.what(), Console::ERROR);
    }

    file.close();
    return result;
}

/* Unit test:
int main() {
    try {
        CSVParser parser;
        auto data = parser.parse("D:\\miscellaneous\\data.csv");

        // Print the parsed data
        for (const auto& row : data) {
            for (const auto& field : row) {
                std::cout << "[" << field << "] ";
            }
            std::cout << "\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
*/