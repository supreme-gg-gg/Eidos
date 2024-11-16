# This is a simple Makefile used to build the example source code.
# This example might requires some modifications in order to work correctly on
# your system.
# If you're not using the Armadillo wrapper, replace `armadillo` with linker commands
# for the BLAS and LAPACK libraries that you are using.

TARGET := mnist_cnn
SRC := mnist_cnn.cpp
LIBS_NAME := armadillo

CXX := g++
CXXFLAGS += -std=c++17
# Use these CXXFLAGS instead if you want to compile with debugging symbols and
# without optimizations.
# CXXFLAGS += -std=c++17 -Wall -Wextra -g -O0
# LDFLAGS  += -fopenmp
LDFLAGS += -L . # /path/to/mlpack/library/  # if installed locally.
# Add header directories for any includes that aren't on the
# default compiler search path.
LDFLAGS += -L/opt/homebrew/Cellar/mlpack/4.5.0/lib/ 
LDFLAGS += -L/opt/homebrew/Cellar/ensmallen/2.21.1/lib/
LDFLAGS += -L/opt/homebrew/Cellar/armadillo/14.0.3/lib/
LDFLAGS += -L/opt/homebrew/Cellar/cereal/1.3.2/lib

INCLFLAGS := -I .
# If you have mlpack or ensmallen installed somewhere nonstandard, uncomment and
# update the lines below.
INCLFLAGS += -I/opt/homebrew/Cellar/mlpack/4.5.0/include/
INCLFLAGS += -I/opt/homebrew/Cellar/ensmallen/2.21.1/include/
INCLFLAGS += -I/opt/homebrew/Cellar/armadillo/14.0.3/include/
INCLFLAGS += -I/opt/homebrew/Cellar/cereal/1.3.2/include
CXXFLAGS  += $(INCLFLAGS)

OBJS := $(SRC:.cpp=.o)
LIBS := $(addprefix -l,$(LIBS_NAME))
CLEAN_LIST := $(TARGET) $(OBJS)

# default rule
default: all

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS) $(LIBS)

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)