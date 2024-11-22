#ifndef CSVPARSER_H
#define CSVPARSER_H

#include <string>
#include <vector>

/*
A basic helper class to parse CSV files
Usage:
```
CSVParser parser();
auto data = parser.parse(filename);
```
*/
class CSVParser {
private:
    std::string filename;
    char delimiter;
    std::vector<std::string> parseLine(const std::string& line);
public:
    CSVParser(char delim);
    std::vector<std::vector<std::string>> parse(const std::string& csvFilename);
};

#endif // CSVPARSER_H