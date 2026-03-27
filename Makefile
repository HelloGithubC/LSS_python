# 编译器设置
CXX := g++
CXXFLAGS := -O2 -Wall -shared -std=c++11 -fPIC -fopenmp
CXXFLAGS_DEBUG := -O0 -Wall -shared -std=c++11 -fPIC -fopenmp -g

# 目录设置
SRC_DIR := src/LSS_python/CPP/src
LIB_DIR := src/LSS_python/CPP/lib

# PYBIND11设置
PYTHON_INCLUDES := $(shell python3-config --includes)
PYBIND11_INCLUDES := -I$(shell python3 -c "import pybind11; print(pybind11.get_include())")
PYBIND11_INCLUDES += -I$(shell python3 -c "import pybind11; print(pybind11.get_include(user=True))")
INCLUDES := $(PYTHON_INCLUDES) $(PYBIND11_INCLUDES)
EXT_SUFFIX := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# 目标文件设置
TARGETS := $(LIB_DIR)/fftpower.so $(LIB_DIR)/mesh.so

# 默认目标
all: $(TARGETS)

# 编译fftpower.so
$(LIB_DIR)/fftpower.so: $(SRC_DIR)/fftpower.cpp $(SRC_DIR)/fftpower.hpp
	@mkdir -p $(LIB_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

# 编译mesh.so
$(LIB_DIR)/mesh.so: $(SRC_DIR)/mesh.cpp $(SRC_DIR)/mesh.hpp
	@mkdir -p $(LIB_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

mesh_pybind: $(SRC_DIR)/mesh_pybind.cpp
	@mkdir -p $(LIB_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $(LIB_DIR)/mesh_pybind$(EXT_SUFFIX)
fftpower_pybind: $(SRC_DIR)/fftpower_pybind.cpp
	@mkdir -p $(LIB_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $(LIB_DIR)/fftpower_pybind$(EXT_SUFFIX)

# 清理目标
clean:
	rm -f $(LIB_DIR)/*.so

# 重新编译
rebuild: clean all

# 伪目标声明
.PHONY: all clean rebuild
