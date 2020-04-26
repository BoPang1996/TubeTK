// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")

at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh, float side_nms_overlap_thresh);

at::Tensor nms(const at::Tensor& dets, const float threshold, const float side_threshold) {
  CHECK_CUDA(dets);
  if (dets.numel() == 0)
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  return nms_cuda(dets, threshold, side_threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
}