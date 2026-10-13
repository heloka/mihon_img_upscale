#ifndef MIHON_QNN_BACKEND_H
#define MIHON_QNN_BACKEND_H

#include <atomic>
#include <array>
#include <cstdint>
#include <string>

namespace qnn_backend {

bool is_runtime_loadable();
int architecture();
std::array<int, 2> platform_info();
bool initialize(const std::string &context_path, int padding,
                bool input_bgr = false);
// Composes, finalizes and serializes a QNN model library (ONNX-converted) in
// this process, then loads the resulting context. Returns false when the QNN
// runtime is unavailable or the library cannot be built.
bool build_from_model_lib(const std::string &model_lib_path,
                          const std::string &out_context_path,
                          int padding, bool input_bgr = false);
bool is_initialized();
int output_scale();
int process_rgba(const uint8_t *input, int width, int height, int input_stride,
                 uint8_t *output, int output_stride,
                 std::atomic<int> *progress,
                 const std::atomic<bool> *should_abort);
void shutdown();

} // namespace qnn_backend

#endif // MIHON_QNN_BACKEND_H
