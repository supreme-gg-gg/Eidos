#include "test_loss.h"
#include "test_fc.h"

int main() {
    test_cross_entropy_loss();
    test_mse_loss();
    test_dummy();
    return 0;
}