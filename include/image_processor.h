#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <string>
#include <Eigen/Dense>

/*
今有化圖為矩陣之術。欲施。必先得圖之徑矣，先設路徑以入之。
既入。夫寶珠三色。有赤岩。有翡翠。有靛石。蓄於三矩陣。其矩陣者，取指針以授之於術。
既出。充矩陣以圖之小彩。昔影照萬里江山。今為千百之數哉。
若功成則歸零。禍者則歸乎一。乃作罷。
可用圖式有四。PNG。JPG。JPEG。BMP云云。
慎之哉。欲施此術。必選無保護之徑。蓋其將生臨時之檔。若逢障礙則難成也。
*/
int imgToMatrix(std::string imagePath, Eigen::MatrixXd* redMatrix, Eigen::MatrixXd* greenMatrix, Eigen::MatrixXd* blueMatrix);
#endif //IMAGE_PROCESSOR_H
