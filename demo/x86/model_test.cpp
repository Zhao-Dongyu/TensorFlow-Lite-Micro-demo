/* Copyright 2023 The tflite Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <math.h>
#include <cstdio>
#include "model_interface.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

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

TfLiteStatus ProfileMemoryAndLatency() {
  tflite::MicroProfiler profiler;
  OpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 1024 * 2000;
  uint8_t tensor_arena[kTensorArenaSize];
  constexpr int kNumResourceVariables = 100;

  tflite::RecordingMicroAllocator* allocator(
      tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
  tflite::RecordingMicroInterpreter interpreter(
      tflite::GetModel(get_model_pointer()), op_resolver, allocator,
      tflite::MicroResourceVariables::Create(allocator, kNumResourceVariables),
      &profiler);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
  TFLITE_CHECK_EQ(interpreter.inputs_size(), 1);
  // interpreter.input(0)->data.f[0] = 1.f;
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  printf(" ");  // Print an empty new line
  profiler.LogTicksPerTagCsv();

  printf(" ");  // Print an empty new line
  interpreter.GetMicroAllocator().PrintAllocations();
  return kTfLiteOk;
}

TfLiteStatus LoadFloatModelAndPerformInference() {
  const tflite::Model* model =
      ::tflite::GetModel(get_model_pointer());
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);
  printf("model->version() = %d\n", model->version());
  OpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 1024 * 2000;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  float input_data[32*32];
  float output_data[10];

  for(int i = 0; i < 32*32; i++) {
    input_data[i] = 1.f;
  }

  TfLiteTensor* input = interpreter.input(0);
  TFLITE_CHECK_NE(input, nullptr);
  TfLiteTensor* output = interpreter.output(0);
  TFLITE_CHECK_NE(output, nullptr);

  float* inTensorData = tflite::GetTensorData<float>(input);
  memcpy(inTensorData, input_data, input->bytes);

  TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  
  memcpy(&output_data[0], tflite::GetTensorData<float>(output), output->bytes);
  printf("output = %f\n", output_data[0]);

  printf("arena_used_bytes = %ld\n", interpreter.arena_used_bytes());
  return kTfLiteOk;
}


int main(int argc, char* argv[]) {
  // tflite::InitializeTarget();
  // TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency());
  TF_LITE_ENSURE_STATUS(LoadFloatModelAndPerformInference());
  printf("~~~ALL TESTS PASSED~~~\n");
  return 0;
}
