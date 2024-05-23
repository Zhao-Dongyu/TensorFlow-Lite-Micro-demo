# 指定编译器
CC = g++
AR = ar

# 指定编译器标志
CFLAGS = -g -Wall -os -DTF_LITE_STATIC_MEMORY

# 增加include目录
INCLUDES = -I../../
INCLUDES += -I../../tensorflow/lite/micro/tools/make/downloads/flatbuffers/include/ 
INCLUDES += -I../../tensorflow/lite/micro/tools/make/downloads/gemmlowp/
INCLUDES +=  -I../include -I../include/dependent_libraries -I./models

# 增加库目录
LDFLAGS = -L.

# 增加库
LDLIBS = -ltensorflow-microlite

# 目标文件
TARGET = test

# 源文件
SRCS = $(wildcard *.cpp) $(wildcard models/*.cpp)

# 对象文件
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET_LIB) $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS) $(LDLIBS)


$(TARGET_LIB):  $(OBJS)
	$(AR) rcs $@ $(OBJS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# 清理目标
.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET) $(TARGET_LIB)
	rm -rf extracted
