# TensorFlow-Lite-Micro-demo

## usage

```shell
cd demo/x86
make
./test
```

本git里面的`libtensorflow-microlite.a`是针对ubuntu编译的，建议自行编译替换。

## blog

### 0 关于tflite micro
![请添加图片描述](https://img-blog.csdnimg.cn/direct/c49dba31bfb84ea58b93ca4ca2976a8d.png)

关于tflite micro在这里接不做过多介绍了，可以查看相关论文：[TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems](https://arxiv.org/abs/2010.08678)

我非常喜欢tflite micro，占用资源很少，是一个用来学习推理框架的好工程。

### 1 克隆仓库
在tensorflow的[git](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro)里面，看到git已经迁移至一个单独的[仓库](https://github.com/tensorflow/tflite-micro)。

> As of June 25, 2021, the TFLM codebase has moved to a stand-alone github repository.

于是执行命令：
`git clone https://github.com/tensorflow/tflite-micro.git`

### 2 编译静态库
求助了ChatGPT 4.0：

>「在这个 https://github.com/tensorflow/tflite-micro ， 怎么编译静态库？」
	
	在TensorFlow Lite Micro的GitHub仓库中，你可以使用以下步骤来编译静态库：
	
	首先，你需要克隆仓库：
	git clone https://github.com/tensorflow/tflite-micro.git
	
	进入克隆下来的仓库目录：
	cd tflite-micro
	
	使用make命令编译静态库：
	make -f tensorflow/lite/micro/tools/make/Makefile TARGET=<TARGET> TARGET_ARCH=<ARCH> microlite
	
	在上述命令中，<TARGET>应替换为你的目标平台（如linux），<ARCH>应替换为你的目标架构（如x86_64）。这将在tensorflow/lite/micro/tools/make/gen/<TARGET>_<ARCH>/lib/目录下生成一个名为libtensorflow-microlite.a的静态库。
	
	请注意，你可能需要根据你的具体需求和环境来修改这些步骤。例如，你可能需要安装额外的依赖库，或者修改makefile来包含你需要的特性。

真是太棒了，于是我执行指令：

`make -f tensorflow/lite/micro/tools/make/Makefile TARGET=linux TARGET_ARCH=x86_64 microlite`

在 make/download/ 目录下，分别下载了flatbuffers、gemmlowp、kissfft、pigweed、ruy(如果你“网络不佳”，这里可能要花些精力)。

编译完成后，得到了静态库文件——
`ar: creating gen/linux_x86_64_default/lib/libtensorflow-microlite.a`

### 3 模型转换
以lenet5模型为例：
1) 先把tflite模型转换为cpp文件：
`xxd -i lenet5.tflite > lenet5.cpp`
2) 包一下模型接口
在lenet5.cpp的文件最后加入了这几行代码
	```c
	unsigned char * get_model_pointer()
	{
	    return lenet5_tflite;
	}
	
	unsigned int get_model_size()
	{
	    return lenet5_tflite_len;
	}
	```
3) 增加函数头文件
	```c
	#ifndef __MODEL_INTERFACE_H__
	#define __MODEL_INTERFACE_H__
	
	unsigned char * get_model_pointer();
	unsigned int get_model_size();
	
	#endif
	```
	这样代码相对比较规范一些，当然也可以直接xxd成头文件直接引用。
	
### 4 编写工程
整个工程比较简单，为了方便引用头文件，我在tflite-micro下新建了一个demo文件夹：
```
.
├── demo
│   └── x86
│       ├── libtensorflow-microlite.a
│       ├── Makefile
│       ├── models
│       │   ├── lenet5.cpp
│       │   ├── lenet5.tflite
│       │   └── model_interface.h
│       ├── model_test.cpp
│       └── test
```

相关工程已经开源至[github](https://github.com/Zhao-Dongyu/TensorFlow-Lite-Micro-demo)，欢迎star，欢迎pr～
### 5 编写demo
#### 5.1 进行算子注册
首先可以看一下模型有哪些算子，以便于确认算子注册类型。（在[netron](https://netron.app/)查看tflite模型）
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/7beead683a304628870fd61864ab9790.png =100x1090)


```c
namespace {
  using OpResolver = tflite::MicroMutableOpResolver<8>;
  TfLiteStatus RegisterOps(OpResolver& op_resolver) {
    TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
    TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
    TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
    TF_LITE_ENSURE_STATUS(op_resolver.AddMaxPool2D());
    TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
    TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
    TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
    TF_LITE_ENSURE_STATUS(op_resolver.AddTanh());
    return kTfLiteOk;
  }
}  // namespace
```
这个过程就是把要用到的算子进行注册。实际上我是缺什么算子加什么就好了。详细过程可以见[算子注册debug过程](#OpResolver)

#### 5.2 推理过程
```c
TfLiteStatus LoadFloatModelAndPerformInference() {
  // get_model_pointer() 送入的就是lenet5的模型指针了
  const tflite::Model* model =
      ::tflite::GetModel(get_model_pointer());
  // 检查模型的版本是否匹配当前的 TFLite 版本。
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);
  // printf("model->version() = %d\n", model->version()); // 好奇的话可以看看版本
  // 创建一个操作符解析器。
  OpResolver op_resolver; 
  // 注册模型中使用的操作符。
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver)); 

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  // 定义一个 2MB 的张量内存区域（tensor_arena），用于解释器分配张量。先往大了写，之后再往小了调
  constexpr int kTensorArenaSize = 1024 * 2000; 
  uint8_t tensor_arena[kTensorArenaSize];
  
  // 创建解释器实例。
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  // 调用 AllocateTensors 方法在 tensor_arena 中分配模型所需的张量内存。
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  float input_data[32*32];
  float output_data[10];

  for(int i = 0; i < 32*32; i++) {
    input_data[i] = 1.f;
  }
  // 获取输入和输出张量的指针，并检查它们是否为空。
  TfLiteTensor* input = interpreter.input(0);
  TFLITE_CHECK_NE(input, nullptr);
  TfLiteTensor* output = interpreter.output(0);
  TFLITE_CHECK_NE(output, nullptr);
  // 将输入数据复制到输入张量中。
  float* inTensorData = tflite::GetTensorData<float>(input);
  memcpy(inTensorData, input_data, input->bytes);
  // 调用 interpreter.Invoke() 执行推理。
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  // 将输出张量的数据复制到 output_data 中，并打印第一个输出值。
  // 当然也可以直接打印 tflite::GetTensorData<float>(output)
  memcpy(&output_data[0], tflite::GetTensorData<float>(output), output->bytes);
  printf("output = %f\n", output_data[0]);
  // 打印使用的内存大小，现在可以根据这个数值去调整 kTensorArenaSize 了。
  printf("arena_used_bytes = %ld\n", interpreter.arena_used_bytes());
  return kTfLiteOk;
}
```
### 6 debug记录
#### 6.1 缺少算子 <a name="OpResolver"></a>
`make`后运行`./test`， 报错：
```shell
Didn't find op for builtin opcode 'TANH'
Failed to get registration from op code TANH
 
Segmentation fault (core dumped)
```
问题很明确，没有进行tanh的算子注册。
具体怎么写呢？在`tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h`这里很容易找到。

#### 6.2 注册表太小
正在一个一个加算子的过程中，遇到这么一个问题：
```shell
Couldn't register builtin op #22, resolver size 
is too small (5).
```
这是因为我定义的数量是5个。
`using OpResolver = tflite::MicroMutableOpResolver<5>;`
把这个增大到算子类型的数量一样就可以了。
这种小细节不注意的话确实容易把人劝退。
#### 6.3 段错误
一旦执行到`interpreter.input(0)->data.f[0] = 1.f;`就段错误。
解决方式：
在makefile里面的`CFLAGS`加` -DTF_LITE_STATIC_MEMORY`
#### 6.4 进一步减小库体积
为了压缩体积，`BUILD_TYPE`使用了`release`进行编译，这期间会遇到MicroPrintf不支持的问题（`release_with_logs`是可以的），进行一些注释就可以。

以及进行`-Os`编译，可以减少很多体积占用。

### 7 实际部署
x86端调试完毕，接下来可以交叉编译tflite micro的库，然后代码移植到另一个工程就好了。

这个过程需要注意一下头文件不要少了。

这个过程可能会遇到诸多问题，欢迎评论交流。

---

相关源码已经开源至[github](https://github.com/Zhao-Dongyu/TensorFlow-Lite-Micro-demo)，欢迎star，欢迎pr～