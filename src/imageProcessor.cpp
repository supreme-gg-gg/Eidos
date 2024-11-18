#include "../include/utils.h"
#include "../include/imageProcessor.h"

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <fstream>
#include <iostream>

int imgToMatrix(std::string imagePath, Eigen::MatrixXd* redMatrix, Eigen::MatrixXd* greenMatrix, Eigen::MatrixXd* blueMatrix) {
    std::filesystem::path imgPath = imagePath;
    if (!std::filesystem::exists(imgPath) || !std::filesystem::is_regular_file(imgPath)) {
        printMsg("\""+imgPath.generic_string()+"\" is not a valid file.", msgFlags::ERROR);
        return 1;
    }
    
    std::string imgExtension = imgPath.extension().generic_string();
    std::string command;
    if (imgExtension == ".jpeg" || imgExtension == ".jpg" || imgPath == ".JPG" || imgPath == ".JPEG") {
        command = "jpegtopnm";
    }
    else if (imgExtension == ".png" || imgExtension == ".PNG") {
        command = "pngtopnm";
    }
    else if (imgExtension == ".bmp" || imgExtension == ".BMP") {
        command = "bmptopnm";
    }
    else {
        printMsg("Unknown image format \""+imgExtension+"\".", msgFlags::ERROR);
        return 1;
    }

    std::filesystem::path working_dir = std::filesystem::current_path();
    std::filesystem::create_directory("temp");
    std::filesystem::current_path(working_dir / "temp");
    command += " -plain "+imgPath.generic_string()+" > "+imgPath.stem().generic_string()+".ppm";
    int sys_res = system(command.c_str());
    if (sys_res == 0) {
        printMsg("Image translated to PNM format for processing.", msgFlags::DEBUG);
    }
    else {
        printMsg("Cannot transform image into PNM for processing", msgFlags::ERROR);
        return 1;
    }
    
    int imgWidth, imgHeight, imgMaxColor;
    std::string imgFormat;
    std::ifstream ppmFile(imgPath.stem().generic_string()+".ppm");
    std::string message;
    if (ppmFile.is_open()) {
        std::string line;
        if (std::getline(ppmFile, line)) {
            std::istringstream iss(line);
            if (!(iss >> imgFormat)) {
                printMsg("Invalid PNM file!", msgFlags::ERROR);
                return 1;
            }
        }
        line = "";
        if (std::getline(ppmFile, line)) {
            std::istringstream iss(line);
            if (!(iss >> imgWidth >> imgHeight)) {
                printMsg("Invalid PNM file!", msgFlags::ERROR);
                return 1;
            }
        }
        if (std::getline(ppmFile, line)) {
            std::istringstream iss(line);
            if (!(iss >> imgMaxColor)) {
                printMsg("Invalid PNM file!", msgFlags::ERROR);
                return 1;
            }
        }
        
        printMsg("Starting to read PNM file content...", msgFlags::DEBUG);
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
        std::string ppmPath = imgPath.stem().generic_string()+".ppm";
        remove(ppmPath.c_str());
        printMsg("Image transformed into matrices successfully!", msgFlags::DEBUG);
        return 0;
    } else {
        printMsg("Unable to open PNM file.", msgFlags::ERROR);
        return 1;
    }
}

/*
int main() {
    std::string ans;
    std::cin >> ans;
    Eigen::MatrixXd R;
    Eigen::MatrixXd G;
    Eigen::MatrixXd B;
    std::cout << imgToMatrix("D:\\downloads\\"+ans,&R, &G, &B);
    std::cout << "\nR:" << std::endl;
    std::cout << R << std::endl;
    std::cout << "\nG: " << std::endl;
    std::cout << G << std::endl;
    std::cout << "\nB: " << std::endl;
    std::cout << B << std::endl;
    return 0;
}
*/