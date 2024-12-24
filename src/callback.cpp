#include "../include/callback.h"
#include "../include/model.h"
#include <iostream>
#include <fstream>


// Some member functions of SaveModel are defined in a separate source file to avoid circular dependencies
void SaveModel::on_epoch_end(int epoch, float loss) {
    if (epoch % save_interval == 0) {
        std::cout << "Saving model at epoch " << epoch << std::endl;
        parent.Serialize(save_path, false);
    }
}

SaveModel* SaveModel::deserialize(std::ifstream& fromFileStream, Model& model) {
    size_t path_size;fromFileStream.read((char*)&path_size, sizeof(size_t));
    char* path_buffer = new char[path_size];
    fromFileStream.read(path_buffer, path_size);
    std::string save_path(path_buffer);
    delete[] path_buffer;
    int save_interval;
    fromFileStream.read((char*)&save_interval, sizeof(int));
    return new SaveModel(model, save_path, save_interval);
}