#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <string>
#include <Eigen/Dense>

/*
Returns 0 upon success.
Takes a path to an image as input parameter.
Writes 3 matrices repholding R,G, and B channels of the image to the three matrix pointer parameters
Supports .jpg, .jpeg, .png, .bmp image formats.
Make sure you call this from a working directory that is not protected, as this function generates intermediate temporary files.
*/
int imgToMatrix(std::string imagePath, Eigen::MatrixXd* redMatrix, Eigen::MatrixXd* greenMatrix, Eigen::MatrixXd* blueMatrix);
#endif //IMAGE_PROCESSOR_H