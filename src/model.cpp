#include "../include/model.h"
#include "../include/console.hpp"
#include "../include/csvparser.h"
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <Eigen/Dense>

namespace fs = std::filesystem;

ImageSample::ImageSample(std::string label, Eigen::MatrixXd& redMatrix, Eigen::MatrixXd& greenMatrix, Eigen::MatrixXd& blueMatrix)
    : label(label),
    data{redMatrix, greenMatrix, blueMatrix}
    {};

/*
今有化圖為矩陣之術。欲施。必先得圖之徑矣，先設路徑以入之。
既入。夫矩陣三則。畜赤青靛三色。其矩陣者，取指針以授之於術。
繩圖之尺寸而使其大小均齊。其參差者。取左上而去其餘。
既出。充矩陣以圖之彩。
若功成則歸零。禍者則歸乎一。書之。乃作罷。
可用圖式有四。PNG。JPG。JPEG。BMP云云。
慎之哉。欲施此術。必選無保護之徑。蓋其將生臨時之檔。若逢障礙則難成也。
*/
int loadImage(fs::path imagePath, Eigen::MatrixXd* redMatrix, Eigen::MatrixXd* greenMatrix, Eigen::MatrixXd* blueMatrix, fs::path workingDir) {
    fs::path imgPath = imagePath;
    if (!fs::exists(imgPath) || !fs::is_regular_file(imgPath)) {
        Console::log("\""+imgPath.generic_string()+"\" is not a valid file.", Console::ERROR);
        return 1;
    }
    
    imgPath = fs::absolute(imgPath);
    
    std::string imgExtension = imgPath.extension().generic_string();
    std::string command[3];
    if (imgExtension == ".jpeg" || imgExtension == ".jpg" || imgPath == ".JPG" || imgPath == ".JPEG") {
        command[0] = "jpegtopnm";
    }
    else if (imgExtension == ".png" || imgExtension == ".PNG") {
        command[0] = "pngtopnm";
    }
    else if (imgExtension == ".bmp" || imgExtension == ".BMP") {
        command[0] = "bmptopnm";
    }
    else {
        Console::log("Unknown image format \""+imgExtension+"\".", Console::ERROR);
        return 1;
    }

    fs::current_path(workingDir);
    fs::path currentDir = fs::current_path();
    fs::create_directory("temp");
    fs::current_path(currentDir / "temp");
    command[0] += " -plain "+imgPath.generic_string()+" > "+imgPath.stem().generic_string()+"-0.ppm 2> nul";
    command[1] = "pamscale -xyfill 256 256 "+imgPath.stem().generic_string()+"-0.ppm > "+imgPath.stem().generic_string()+"-1.ppm";
    command[2] = "pamcut -width 256 -height 256 "+imgPath.stem().generic_string()+"-1.ppm > "+imgPath.stem().generic_string()+".ppm";
    for (int i = 0; i < 3; i++) {
        if (system(command[i].c_str()) != 0) {
            Console::log("Failed to transform image into PNM for processing", Console::ERROR);
            return 1;
        };
    }
    Console::log("Image translated to PNM format for processing.", Console::DEBUG);
    
    int imgWidth, imgHeight, imgMaxColor;
    std::string imgFormat;
    std::ifstream ppmFile(imgPath.stem().generic_string()+".ppm");
    std::string message;
    if (ppmFile.is_open()) {
        std::string line;
        if (std::getline(ppmFile, line)) {
            std::istringstream iss(line);
            if (!(iss >> imgFormat)) {
                Console::log("Invalid PNM file!", Console::ERROR);
                return 1;
            }
        }
        line = "";
        if (std::getline(ppmFile, line)) {
            std::istringstream iss(line);
            if (!(iss >> imgWidth >> imgHeight)) {
                Console::log("Invalid PNM file!", Console::ERROR);
                return 1;
            }
        }
        if (std::getline(ppmFile, line)) {
            std::istringstream iss(line);
            if (!(iss >> imgMaxColor)) {
                Console::log("Invalid PNM file!", Console::ERROR);
                return 1;
            }
        }
        
        Console::log("Starting to read PNM file content...", Console::DEBUG);
        redMatrix->resize(imgHeight, imgWidth);
        greenMatrix->resize(imgHeight, imgWidth);
        blueMatrix->resize(imgHeight, imgWidth);
        int col = 0;
        int row = 0;
        while (std::getline(ppmFile, line)) {
            std::istringstream iss(line);
            int buffer[3];
            while (iss >> buffer[0] >> buffer[1] >> buffer[2]) {
                (*redMatrix)(row, col) = buffer[0];
                (*greenMatrix)(row, col) = buffer[1];
                (*blueMatrix)(row, col) = buffer[2];
                col++;
                if (col >= imgWidth) {
                    col = 0;
                    row++;
                }
            }
        }
        
        ppmFile.close();
        std::string ppm0Path = imgPath.stem().generic_string()+"-0.ppm";
        std::string ppm1Path = imgPath.stem().generic_string()+"-1.ppm";
        std::string ppmPath = imgPath.stem().generic_string()+".ppm";
        remove(ppm0Path.c_str());
        remove(ppm1Path.c_str());
        remove(ppmPath.c_str());
        fs::current_path(currentDir);
        
        Console::log("Image transformed into matrices successfully!", Console::DEBUG);
        return 0;
    } else {
        Console::log("Unable to open PNM file.", Console::ERROR);
        return 1;
    }
}

std::string CNN_Model::Info() {
    std::string output = "";
    output += "Number of layers: "+std::to_string(this->parameters.size())+"\n";
    output += "Training sample size: "+std::to_string(this->trainingSamples.size())+"\n";
    output += "Testing sample size: "+std::to_string(this->testingSamples.size());
    return output;
}

int CNN_Model::loadData(std::string dataLabelsPath, float ftrainToTestSplitRatio) {
    Console::log("Loading labelled data from CSV... (This may take a while.)");
    CSVParser parser;
    auto csvData = parser.parse(dataLabelsPath);
    int colNum = csvData[0].size();
    if (colNum != 2) {
        Console::log("Invalid label file.\nExpected syntax:\n"
            "1| label,data"
            "2| <label>,<path/to/image>"
            "3| <label>,<path/to/image>"
            "..."
        , Console::ERROR);
        return 1;
    }
    
    // Temporary vector for randomizing traning data
    std::vector<int> shuffledImgPaths;
    
    fs::path currentDir = fs::current_path();
    fs::current_path(fs::absolute(dataLabelsPath).parent_path());
    Console::log(std::to_string(csvData.size()), Console::DEBUG);
    for (int i = 1; i < csvData.size(); i++) {
        if (csvData[i].size() != colNum) {
            Console::log("Invalid label file.\nExpected syntax:\n"
                "1| label,data"
                "2| <label>,<path/to/image>"
                "3| <label>,<path/to/image>"
                "..."
            , Console::ERROR);
            return 1;
        }
        fs::path imgPath = csvData[i][1];
        if (fs::is_regular_file(imgPath)) {
            std::string extension = imgPath.extension().generic_string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp") {
                if (!csvData[i][0].empty()) {
                    shuffledImgPaths.push_back(i);
                }
                else {
                    Console::log("Bad entry in data labels: In entry "+std::to_string(i)+": Empty label. Skipped.", Console::WARNING);
                }
            }
            else {
                Console::log("Bad entry in data labels: In entry "+std::to_string(i)+": Unsupported image format \""+extension+"\". Skipped.", Console::WARNING);
            }
        }
        else {
            Console::log("Bad entry in data labels: In entry "+std::to_string(i)+": \""+imgPath.generic_string()+"\" is not a valid file. Skipped.", Console::WARNING);
        }
    }
    
    std::random_device rd;
    std::mt19937 g(rd()); 
    std::shuffle(shuffledImgPaths.begin(), shuffledImgPaths.end(), g);
    int i;
    for (i = 0; i < shuffledImgPaths.size()*ftrainToTestSplitRatio; i++) {
        Eigen::MatrixXd red;
        Eigen::MatrixXd green;
        Eigen::MatrixXd blue;
        fs::current_path(fs::absolute(dataLabelsPath).parent_path());
        if (loadImage(fs::absolute(csvData[shuffledImgPaths[i]][1]), &red, &green, &blue, currentDir) != 0) return 1;
        this->trainingSamples.push_back(ImageSample(csvData[shuffledImgPaths[i]][0], red, green, blue));
    }for (; i < shuffledImgPaths.size(); i++) {
        Eigen::MatrixXd red;
        Eigen::MatrixXd green;
        Eigen::MatrixXd blue;
        fs::current_path(fs::absolute(dataLabelsPath).parent_path());
        if (loadImage(fs::absolute(csvData[shuffledImgPaths[i]][1]), &red, &green, &blue, currentDir) != 0) return 1;
        this->testingSamples.push_back(ImageSample(csvData[shuffledImgPaths[i]][0], red, green, blue));
    }
    fs::current_path(currentDir);
    Console::log("All image data loaded successfully!");
    return 0;
}

int CNN_Model::Train() {
    return 1;
}
int CNN_Model::Test() {
    return 1;
}
int CNN_Model::serializeParameters(std::string toFilePath) {
    return 1;
}
int CNN_Model::deserializeParameters(std::string fromFilePath) {
    return 1;
}

/* Unit test:
int main() {
    Console::config(false);
    CNN_Model model;
    std::string input;
    std::cout << "Enter path to data labels CSV file: ";
    std::cin >> input;
    model.loadData(input, 2.0/3);
    Console::log(model.Info());
    return 0;
}
*/