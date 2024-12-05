#include <gtest/gtest.h>

// Function to be tested
int add(int a, int b) {
    return a + b;
}

// Test case for the add function
TEST(AdditionTest, HandlesPositiveInput) {
    EXPECT_EQ(add(1, 2), 3);
    EXPECT_EQ(add(10, 20), 30);
}

TEST(AdditionTest, HandlesNegativeInput) {
    EXPECT_EQ(add(-1, -2), -3);
    EXPECT_EQ(add(-10, -20), -30);
}

TEST(AdditionTest, HandlesMixedInput) {
    EXPECT_EQ(add(-1, 2), 1);
    EXPECT_EQ(add(10, -20), -10);
}

// Main function to run all the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}