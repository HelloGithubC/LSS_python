# 编译器设置
CXX := g++
CXXFLAGS := -O2 -Wall -shared -std=c++11 -fPIC -fopenmp

# 目录设置
SRC_DIR := ./src
LIB_DIR := ./lib

# 目标文件设置
TARGETS := $(LIB_DIR)/fftpower.so $(LIB_DIR)/mesh.so

# 默认目标
all: $(TARGETS)

# 编译fftpower.so
$(LIB_DIR)/fftpower.so: $(SRC_DIR)/fftpower.cpp
	@mkdir -p $(LIB_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

# 编译mesh.so
$(LIB_DIR)/mesh.so: $(SRC_DIR)/mesh.cpp
	@mkdir -p $(LIB_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

# 清理目标
clean:
	rm -rf $(LIB_DIR)

# 重新编译
rebuild: clean all

# 伪目标声明
.PHONY: all clean rebuild
