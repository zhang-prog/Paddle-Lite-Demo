// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h" // NOLINT
#include <fstream>      // NOLINT
#include <iostream>     // NOLINT
#include <sys/time.h>   // NOLINT
#include <time.h>       // NOLINT
#include <vector>       // NOLINT
/////////////////////////////////////////////////////////////////////////
// If this demo is linked to static library: libpaddle_api_light_bundled.a
// , you should include `paddle_use_ops.h` and `paddle_use_kernels.h` to
// avoid linking errors such as `unsupport ops or kernels`.
/////////////////////////////////////////////////////////////////////////
// #include "paddle_use_kernels.h"  // NOLINT
// #include "paddle_use_ops.h"      // NOLINT
using namespace paddle::lite_api; // NOLINT

template <typename T>
void get_value_from_sstream(std::stringstream *ss, T *value) {
  (*ss) >> (*value);
}

template <>
void get_value_from_sstream<std::string>(std::stringstream *ss,
                                         std::string *value) {
  *value = ss->str();
}

template <typename T>
std::vector<T> split_string(const std::string &str, char sep) {
  std::stringstream ss;
  std::vector<T> values;
  T value;
  values.clear();
  for (auto c : str) {
    if (c != sep) {
      ss << c;
    } else {
      get_value_from_sstream<T>(&ss, &value);
      values.push_back(std::move(value));
      ss.str({});
      ss.clear();
    }
  }
  if (!ss.str().empty()) {
    get_value_from_sstream<T>(&ss, &value);
    values.push_back(std::move(value));
    ss.str({});
    ss.clear();
  }
  return values;
}

typedef struct {
  int width;
  int height;
  std::vector<float> mean;
  std::vector<float> std;
  float draw_weight{0.5f};
} CONFIG;

bool read_file(const std::string &filename,
               std::vector<char> *contents,
               bool binary = true) {
  FILE *fp = fopen(filename.c_str(), binary ? "rb" : "r");
  if (!fp) return false;
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  contents->clear();
  contents->resize(size);
  size_t offset = 0;
  char *ptr = reinterpret_cast<char *>(&(contents->at(0)));
  while (offset < size) {
    size_t already_read = fread(ptr, 1, size - offset, fp);
    offset += already_read;
    ptr += already_read;
  }
  fclose(fp);
  return true;
}

std::vector<cv::Scalar> generate_color_map(int num_classes) {
  if (num_classes < 10) {
    num_classes = 10;
  }
  std::vector<cv::Scalar> color_map = std::vector<cv::Scalar>(num_classes);
  for (int i = 0; i < num_classes; i++) {
    int j = 0;
    int label = i;
    int R = 0, G = 0, B = 0;
    while (label) {
      R |= (((label >> 0) & 1) << (7 - j));
      G |= (((label >> 1) & 1) << (7 - j));
      B |= (((label >> 2) & 1) << (7 - j));
      j++;
      label >>= 3;
    }
    color_map[i] = cv::Scalar(R, G, B);
  }
  return color_map;
}

void load_labels(const std::string &path, std::vector<std::string> *labels) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    std::cerr << "Load input label file error." << std::endl;
    exit(1);
  }
  std::string line;
  while (getline(ifs, line)) {
    labels->push_back(line);
  }
  ifs.close();
}

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

// fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
void neon_mean_scale(const float *din, float *dout, int size, float *mean,
                     float *scale) {
  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(1.f / scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(1.f / scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(1.f / scale[2]);

  float *dout_c0 = dout;
  float *dout_c1 = dout + size;
  float *dout_c2 = dout + size * 2;

  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(din);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dout_c0, vs0);
    vst1q_f32(dout_c1, vs1);
    vst1q_f32(dout_c2, vs2);

    din += 12;
    dout_c0 += 4;
    dout_c1 += 4;
    dout_c2 += 4;
  }
  for (; i < size; i++) {
    *(dout_c0++) = (*(din++) - mean[0]) / scale[0];
    *(dout_c0++) = (*(din++) - mean[1]) / scale[1];
    *(dout_c0++) = (*(din++) - mean[2]) / scale[2];
  }
}

void pre_process(std::shared_ptr<PaddlePredictor> predictor,
                 const std::string img_path, int width, int height) {
  // Prepare input data from image
  float scale[3] = {1.f, 1.f, 1.f};
  float mean[3] = {0.5f, 0.5f, 0.5f};
  std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
  input_tensor->Resize({1, 3, height, width});
  // read img and pre-process
  cv::Mat origin_image = cv::imread(img_path);
  cv::Mat resized_image;
  cv::resize(origin_image,
              resized_image,
              cv::Size(width, height),
              0,
              0);
  cv::Mat rgb_image;
  if (resized_image.channels() == 3) {
    cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);
  } else if (resized_image.channels() == 4) {
    cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGRA2RGB);
  } else {
    printf("The channel size should be 4 or 3, but receive %d!\n",
            resized_image.channels());
    exit(-1);
  }
  rgb_image.convertTo(rgb_image, CV_32FC3, 1 / 255.f);

  auto *image_data = input_tensor->mutable_data<float>();
  neon_mean_scale(reinterpret_cast<const float *>(rgb_image.data), image_data, width * height, mean, scale);
}

void post_process(std::shared_ptr<PaddlePredictor> predictor,
                  const std::string img_path,
                  const std::vector<std::string> &labels,
                  int width,
                  int height) {
  auto color_map = generate_color_map(1000);
  auto output_tensor = predictor->GetOutput(0);
  auto output_shape = output_tensor->shape();
  auto output_rank = output_shape.size();
  // image tensor
  cv::Mat origin_image = cv::imread(img_path);
  cv::Mat resized_image;
  cv::resize(origin_image,
              resized_image,
              cv::Size(width, height),
              0,
              0);
  int64_t output_height, output_width;
  std::vector<int64_t> mask_data;
  if (output_rank == 3) {
    output_height = output_shape[1];
    output_width = output_shape[2];
    auto image_size = output_height * output_width;
    mask_data.resize(image_size);
    if (output_tensor->precision() == PRECISION(kInt32)) {
      auto output_data = output_tensor->data<int32_t>();
      for (int64_t j = 0; j < image_size; j++) {
        mask_data[j] = static_cast<int64_t>(output_data[j]);
      }
    } else if (output_tensor->precision() == PRECISION(kInt64)) {
      auto output_data = output_tensor->data<int64_t>();
      memcpy(mask_data.data(), output_data, image_size * sizeof(int64_t));
    }
  } else if (output_rank == 4) {
    auto num_classes = output_shape[1];
    output_height = output_shape[2];
    output_width = output_shape[3];
    auto image_size = output_height * output_width;
    mask_data.resize(image_size);
    auto output_data = output_tensor->data<float>();
    for (int64_t j = 0; j < image_size; j++) {
      auto max_value = output_data[j];
      auto max_index = 0;
      for (int64_t k = 1; k < num_classes; k++) {
        auto cur_value = output_data[k * image_size + j];
        if (max_value < cur_value) {
          max_value = cur_value;
          max_index = k;
        }
      }
      mask_data[j] = max_index;
    }
  } else {
    printf(
        "The rank of the output tensor should be 3 or 4, but receive %d!\n",
        int(output_rank));
    exit(-1);
  }
  auto mask_image = resized_image.clone();
  int64_t mask_index = 0;
  for (int64_t j = 0; j < output_height; j++) {
    for (int64_t k = 0; k < output_width; k++) {
      auto class_id = mask_data[mask_index++];
      mask_image.at<cv::Vec3b>(j, k) =
          cv::Vec3b(color_map[class_id][2],
                    color_map[class_id][1],
                    color_map[class_id][0]);  // RGB->BGR
    }
  }
  cv::resize(mask_image,
              mask_image,
              cv::Size(origin_image.cols, origin_image.rows),
              0,
              0);
  cv::addWeighted(origin_image,
                  0.5,
                  mask_image,
                  0.5,
                  0,
                  origin_image);
  std::string output_path = "result.jpg";
  cv::imwrite(output_path, origin_image);
}

void run_model(std::string model_file, std::string img_path,
               const std::vector<std::string> &labels, int width, int height,
               int power_mode, int thread_num, int repeats, int warmup) {
  // 1. Set MobileConfig
  MobileConfig config;
  config.set_model_from_file(model_file);
  config.set_power_mode(static_cast<paddle::lite_api::PowerMode>(power_mode));
  config.set_threads(thread_num);

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);

  // 3. Prepare input data from image
  pre_process(predictor, img_path, width, height);

  // 4. Run predictor
  double first_duration{-1};
  for (size_t widx = 0; widx < warmup; ++widx) {
    if (widx == 0) {
      auto start = GetCurrentUS();
      predictor->Run();
      first_duration = (GetCurrentUS() - start) / 1000.0;
    } else {
      predictor->Run();
    }
  }

  double sum_duration = 0.0;
  double max_duration = 1e-5;
  double min_duration = 1e5;
  double avg_duration = -1;
  for (size_t ridx = 0; ridx < repeats; ++ridx) {
    auto start = GetCurrentUS();

    predictor->Run();

    auto duration = (GetCurrentUS() - start) / 1000.0;
    sum_duration += duration;
    max_duration = duration > max_duration ? duration : max_duration;
    min_duration = duration < min_duration ? duration : min_duration;
    if (first_duration < 0) {
      first_duration = duration;
    }
  }

  avg_duration = sum_duration / static_cast<float>(repeats);
  std::cout << "\n======= benchmark summary =======\n"
            << "input_shape(s) (NCHW): {1, 3, " << height << ", " << width
            << "}\n"
            << "model_dir:" << model_file << "\n"
            << "warmup:" << warmup << "\n"
            << "repeats:" << repeats << "\n"
            << "power_mode:" << power_mode << "\n"
            << "thread_num:" << thread_num << "\n"
            << "*** time info(ms) ***\n"
            << "1st_duration:" << first_duration << "\n"
            << "max_duration:" << max_duration << "\n"
            << "min_duration:" << min_duration << "\n"
            << "avg_duration:" << avg_duration << "\n";

  // 5. Get output and post process
  std::cout << "\n====== output summary ====== " << std::endl;
  post_process(predictor, img_path, labels, width, height);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "[ERROR] usage: " << argv[0]
              << " model_file image_path label_file\n";
    exit(1);
  }
  std::cout << "This parameters are optional: \n"
            << " <input_width>, eg: 224 \n"
            << " <input_height>, eg: 224 \n"
            << "  <power_mode>, 0: big cluster, high performance\n"
               "                1: little cluster\n"
               "                2: all cores\n"
               "                3: no bind\n"
            << "  <thread_num>, eg: 1 for single thread \n"
            << "  <repeats>, eg: 100\n"
            << "  <warmup>, eg: 10\n"
            << std::endl;
  std::string model_file = argv[1];
  std::string img_path = argv[2];
  std::string label_file = argv[3];
  std::vector<std::string> labels;
  load_labels(label_file, &labels);
  int height = 512;
  int width = 1024;
  int warmup = 0;
  int repeats = 1;
  int power_mode = 0;
  int thread_num = 1;
  int use_gpu = 0;
  if (argc > 5) {
    width = atoi(argv[4]);
    height = atoi(argv[5]);
  }
  if (argc > 6) {
    thread_num = atoi(argv[6]);
  }
  if (argc > 7) {
    power_mode = atoi(argv[7]);
  }
  if (argc > 8) {
    repeats = atoi(argv[8]);
  }
  if (argc > 9) {
    warmup = atoi(argv[9]);
  }
  if (argc > 10) {
    use_gpu = atoi(argv[10]);
  }

  run_model(model_file, img_path, labels, width, height, power_mode, thread_num,
            repeats, warmup);
  return 0;
}
