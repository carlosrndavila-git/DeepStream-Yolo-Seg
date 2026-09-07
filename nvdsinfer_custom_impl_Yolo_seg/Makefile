CUDA_VER?=
ifeq ($(CUDA_VER),)
	$(error "CUDA_VER is not set")
endif

CUDA_VER_MAJOR:= $(word 1,$(subst ., ,$(CUDA_VER)))
CUDA_VER_MINOR:= $(word 2,$(subst ., ,$(CUDA_VER)))
CUDA_VER_NUM:= $(CUDA_VER_MAJOR)$(CUDA_VER_MINOR)

CUDA_ARCH:= 53 60 61 62 70 72 75
ifeq ($(shell expr $(CUDA_VER_NUM) \>= 110),1)
  CUDA_ARCH+= 80
endif
ifeq ($(shell expr $(CUDA_VER_NUM) \>= 111),1)
  CUDA_ARCH+= 86
endif
ifeq ($(shell expr $(CUDA_VER_NUM) \>= 118),1)
  CUDA_ARCH+= 87 89
endif
ifeq ($(shell expr $(CUDA_VER_NUM) \>= 120),1)
  CUDA_ARCH+= 90
endif
ifeq ($(shell expr $(CUDA_VER_NUM) \>= 128),1)
  CUDA_ARCH+= 100 103 110 120
endif

GENCODE_FLAGS:= $(foreach a,$(CUDA_ARCH),-gencode arch=compute_$(a),code=sm_$(a) \
				-gencode arch=compute_$(a),code=compute_$(a))

SM?=
ifneq ($(SM),)
	GENCODE_FLAGS:= -gencode arch=compute_$(SM),code=sm_$(SM) -gencode arch=compute_$(SM),code=compute_$(SM)
endif

CXX:= g++
NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc

TARGET_LIB:= libnvdsinfer_custom_impl_Yolo_seg.so

CFLAGS:= -Wall -std=c++17 -shared -fPIC -Wno-error=deprecated-declarations

CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include
CFLAGS+= $(shell pkg-config --cflags opencv4)
CUFLAGS:= -I/opt/nvidia/deepstream/deepstream/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include

LIBS+= -lnvinfer -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lstdc++fs
LIBS+= -lopencv_core -lopencv_imgproc

LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

SRCS:= $(wildcard *.cpp)
SRCS+= $(wildcard trt_plugins/*.cpp)
SRCS+= $(wildcard trt_plugins/efficientNMSPlugin/*.cpp)
SRCS+= $(wildcard trt_plugins/efficientNMSPlugin/*.cu)
SRCS+= $(wildcard trt_plugins/roiAlignPlugin/*.cpp)
SRCS+= $(wildcard trt_plugins/roiAlignPlugin/*.cu)

INCS:= $(wildcard *.h)
INCS+= $(wildcard trt_plugins/*.h)
INCS+= $(wildcard trt_plugins/efficientNMSPlugin/*.h)
INCS+= $(wildcard trt_plugins/efficientNMSPlugin/*.cuh)
INCS+= $(wildcard trt_plugins/roiAlignPlugin/*.h)
INCS+= $(wildcard trt_plugins/roiAlignPlugin/*.cuh)

OBJS:= $(addsuffix .o, $(basename $(SRCS)))

all: $(TARGET_LIB)

%.o: %.cpp Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ $(GENCODE_FLAGS) --compiler-options '-fPIC' $(CUFLAGS) $<

$(TARGET_LIB) : $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LFLAGS)

clean:
	rm -rf $(OBJS) $(TARGET_LIB)
