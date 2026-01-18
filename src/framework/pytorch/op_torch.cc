#include <torch/extension.h>
#include "framework/op.h"
#include "base/tensor.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

// Log level: 0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG
#include <glog/logging.h>

namespace {
struct LogInit {
  LogInit() { recstore::ConfigureLogging(); }
};
static LogInit log_init_instance;
} // namespace

namespace recstore {
namespace framework {

static inline base::RecTensor
ToRecTensor(const torch::Tensor& tensor, base::DataType dtype) {
  std::vector<int64_t> shape;
  for (int i = 0; i < tensor.dim(); ++i) {
    shape.push_back(tensor.size(i));
  }
  return base::RecTensor(const_cast<void*>(tensor.data_ptr()), shape, dtype);
}

torch::Tensor emb_read_torch(const torch::Tensor& keys, int64_t embedding_dim) {
  recstore::ConfigureLogging();
  LOG(INFO) << "keys shape=" << keys.sizes() << ", dtype=" << keys.dtype()
            << ", data_ptr=" << keys.data_ptr();
  LOG(INFO) << "embedding_dim=" << embedding_dim;
  torch::Tensor cpu_keys = keys;
  if (keys.is_cuda()) {
    LOG(INFO) << "copying GPU keys to CPU";
    cpu_keys = keys.cpu();
  }
  if (cpu_keys.size(0) > 0) {
    auto cpu_keys_acc = cpu_keys.accessor<int64_t, 1>();
    std::ostringstream oss;
    oss << "keys start with: ";
    for (int i = 0; i < std::min((int64_t)10, keys.size(0)); ++i)
      oss << cpu_keys_acc[i] << ", ";
    LOG(INFO) << oss.str();
  }
  LOG(INFO) << "emb_read_torch called: keys shape=" << cpu_keys.sizes()
            << ", dtype=" << cpu_keys.dtype()
            << ", embedding_dim=" << embedding_dim;
  TORCH_CHECK(cpu_keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(cpu_keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(cpu_keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(embedding_dim > 0, "Embedding dimension must be positive");

  const int64_t num_keys = cpu_keys.size(0);
  if (num_keys == 0) {
    LOG(INFO) << "num_keys==0, returning empty";
    return torch::empty(
        {0, embedding_dim}, cpu_keys.options().dtype(torch::kFloat32));
  }

  auto op = GetKVClientOp();

  auto values = torch::empty(
      {num_keys, embedding_dim}, keys.options().dtype(torch::kFloat32));
  torch::Tensor cpu_values = values;
  if (values.is_cuda()) {
    LOG(INFO) << "copying GPU values to CPU for C++ operation";
    cpu_values = values.cpu();
  }
  TORCH_CHECK(cpu_values.is_contiguous(),
              "Internal error: Created values tensor is not contiguous");
  LOG(INFO) << "values shape=" << cpu_values.sizes() << ", dtype="
            << cpu_values.dtype() << ", data_ptr=" << cpu_values.data_ptr();
  if (cpu_values.size(0) > 0) {
    auto values_acc = cpu_values.accessor<float, 2>();
    std::ostringstream oss;
    oss << "values start with: ";
    for (int i = 0; i < std::min((int64_t)10, cpu_values.size(0)); ++i) {
      oss << "[";
      for (int j = 0; j < std::min((int64_t)10, cpu_values.size(1)); ++j) {
        oss << values_acc[i][j] << ", ";
      }
      oss << "] ";
    }
    LOG(INFO) << oss.str();
  }

  base::RecTensor rec_keys   = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_values = ToRecTensor(cpu_values, base::DataType::FLOAT32);

  LOG(INFO) << "calling op->EmbRead";
  op->EmbRead(rec_keys, rec_values);
  LOG(INFO) << "EmbRead done";

  if (values.is_cuda()) {
    LOG(INFO) << "copying results back to GPU";
    values.copy_(cpu_values);
  }

  return values;
}

// Async prefetch: returns a unique prefetch id (uint64_t)
int64_t emb_prefetch_torch(const torch::Tensor& keys) {
  recstore::ConfigureLogging();
  // LOG(INFO) << "emb_prefetch_torch called";
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");

  auto op                = GetKVClientOp();
  torch::Tensor cpu_keys = keys;
  if (keys.is_cuda()) {
    cpu_keys = keys.cpu();
  }
  base::RecTensor rec_keys = ToRecTensor(cpu_keys, base::DataType::UINT64);
  // Dummy values tensor (unused by backend prefetch implementation)
  auto dummy_vals = torch::empty({0, 0}, keys.options().dtype(torch::kFloat32));
  base::RecTensor rec_vals = ToRecTensor(dummy_vals, base::DataType::FLOAT32);
  uint64_t pid             = op->EmbPrefetch(rec_keys, rec_vals);
  return static_cast<int64_t>(pid);
}

// Wait for prefetch and return result tensor [N, embedding_dim] on CPU
torch::Tensor
emb_wait_result_torch(int64_t prefetch_id, int64_t embedding_dim) {
  recstore::ConfigureLogging();
  LOG(INFO) << "emb_wait_result_torch called: pid=" << prefetch_id
            << ", dim=" << embedding_dim;
  TORCH_CHECK(embedding_dim > 0, "Embedding dimension must be positive");
  auto op = GetKVClientOp();
  op->WaitForPrefetch(static_cast<uint64_t>(prefetch_id));
  std::vector<std::vector<float>> vecs;
  op->GetPretchResult(static_cast<uint64_t>(prefetch_id), &vecs);
  const int64_t L = static_cast<int64_t>(vecs.size());
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto out = torch::empty({L, embedding_dim}, options);
  if (L > 0) {
    float* dst = out.data_ptr<float>();
    for (int64_t i = 0; i < L; ++i) {
      const auto& row = vecs[i];
      if (!row.empty()) {
        const int64_t copy_d =
            std::min<int64_t>(embedding_dim, static_cast<int64_t>(row.size()));
        std::memcpy(
            dst + i * embedding_dim, row.data(), sizeof(float) * copy_d);
        if (copy_d < embedding_dim) {
          std::memset(dst + i * embedding_dim + copy_d,
                      0,
                      sizeof(float) * (embedding_dim - copy_d));
        }
      } else {
        std::memset(dst + i * embedding_dim, 0, sizeof(float) * embedding_dim);
      }
    }
  }
  return out;
}

void emb_update_torch(const torch::Tensor& keys, const torch::Tensor& grads) {
  throw std::runtime_error(
      "emb_update_torch is deprecated. Use the Python-based sparse optimizer.");
}

void emb_update_table_torch(const std::string& table_name,
                            const torch::Tensor& keys,
                            const torch::Tensor& grads) {
  recstore::ConfigureLogging();
  LOG(INFO) << "emb_update_table_torch called for table=" << table_name
            << ", keys shape=" << keys.sizes()
            << ", grads shape=" << grads.sizes();
  TORCH_CHECK(!table_name.empty(), "table_name must be non-empty");
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");

  TORCH_CHECK(grads.dim() == 2, "Grads tensor must be 2-dimensional");
  TORCH_CHECK(grads.scalar_type() == torch::kFloat32,
              "Grads tensor must have dtype float32");
  TORCH_CHECK(grads.is_contiguous(), "Grads tensor must be contiguous");
  TORCH_CHECK(keys.size(0) == grads.size(0),
              "Keys and grads tensors must have the same number of entries");

  if (keys.size(0) == 0) {
    LOG(INFO) << "emb_update_table_torch: num_keys==0, early return";
    return;
  }

  auto op = GetKVClientOp();

  torch::Tensor cpu_keys  = keys;
  torch::Tensor cpu_grads = grads;
  if (keys.is_cuda()) {
    LOG(INFO) << "emb_update_table_torch: copying GPU keys to CPU";
    cpu_keys = keys.cpu();
  }
  if (grads.is_cuda()) {
    LOG(INFO) << "emb_update_table_torch: copying GPU grads to CPU";
    cpu_grads = grads.cpu();
  }

  base::RecTensor rec_keys  = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_grads = ToRecTensor(cpu_grads, base::DataType::FLOAT32);

  LOG(INFO) << "emb_update_table_torch: calling op->EmbUpdate";
  op->EmbUpdate(table_name, rec_keys, rec_grads);
  LOG(INFO) << "emb_update_table_torch: EmbUpdate done";
}

bool init_embedding_table_torch(const std::string& table_name,
                                int64_t num_embeddings,
                                int64_t embedding_dim) {
  recstore::ConfigureLogging();
  LOG(INFO) << "init_embedding_table_torch called for table=" << table_name
            << ", num_embeddings=" << num_embeddings
            << ", embedding_dim=" << embedding_dim;
  TORCH_CHECK(!table_name.empty(), "table_name must be non-empty");
  TORCH_CHECK(num_embeddings > 0, "num_embeddings must be positive");
  TORCH_CHECK(embedding_dim > 0, "embedding_dim must be positive");

  EmbeddingTableConfig cfg{};
  cfg.num_embeddings = static_cast<uint64_t>(num_embeddings);
  cfg.embedding_dim  = static_cast<uint64_t>(embedding_dim);

  auto op = GetKVClientOp();
  return op->InitEmbeddingTable(table_name, cfg);
}

void emb_write_torch(const torch::Tensor& keys, const torch::Tensor& values) {
  recstore::ConfigureLogging();
  LOG(INFO) << "emb_write_torch: keys shape=" << keys.sizes()
            << ", dtype=" << keys.dtype() << ", data_ptr=" << keys.data_ptr();
  LOG(INFO) << "emb_write_torch: values shape=" << values.sizes() << ", dtype="
            << values.dtype() << ", data_ptr=" << values.data_ptr();
  if (keys.size(0) > 0) {
    auto keys_acc = keys.accessor<int64_t, 1>();
    std::ostringstream oss;
    oss << "emb_write_torch: keys start with: ";
    for (int i = 0; i < std::min((int64_t)10, keys.size(0)); ++i)
      oss << keys_acc[i] << ", ";
    LOG(INFO) << oss.str();
  }
  if (values.size(0) > 0) {
    auto values_acc = values.accessor<float, 2>();
    std::ostringstream oss;
    oss << "emb_write_torch: values start with: ";
    for (int i = 0; i < std::min((int64_t)10, values.size(0)); ++i) {
      oss << "[";
      for (int j = 0; j < std::min((int64_t)10, values.size(1)); ++j) {
        oss << values_acc[i][j] << ", ";
      }
      oss << "] ";
    }
    LOG(INFO) << oss.str();
  }
  LOG(INFO) << "emb_write_torch called: keys shape=" << keys.sizes()
            << ", values shape=" << values.sizes();
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");
  TORCH_CHECK(values.dim() == 2, "Values tensor must be 2-dimensional");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32,
              "Values tensor must have dtype float32");
  TORCH_CHECK(values.is_contiguous(), "Values tensor must be contiguous");
  TORCH_CHECK(keys.size(0) == values.size(0),
              "Keys and Values tensors must have the same number of entries");

  if (keys.size(0) == 0) {
    LOG(INFO) << "emb_write_torch: num_keys==0, early return";
    return;
  }

  auto op = GetKVClientOp();

  torch::Tensor cpu_keys   = keys;
  torch::Tensor cpu_values = values;
  if (keys.is_cuda()) {
    LOG(INFO) << "emb_write_torch: copying GPU keys to CPU";
    cpu_keys = keys.cpu();
  }
  if (values.is_cuda()) {
    LOG(INFO) << "emb_write_torch: copying GPU values to CPU";
    cpu_values = values.cpu();
  }

  base::RecTensor rec_keys   = ToRecTensor(cpu_keys, base::DataType::UINT64);
  base::RecTensor rec_values = ToRecTensor(cpu_values, base::DataType::FLOAT32);

  LOG(INFO) << "emb_write_torch: calling op->EmbWrite";
  op->EmbWrite(rec_keys, rec_values);
  LOG(INFO) << "emb_write_torch: EmbWrite done";
}

void set_ps_config_torch(const std::string& host, int64_t port) {
  recstore::ConfigureLogging();
  LOG(INFO) << "set_ps_config_torch called: host=" << host << " port=" << port;
  auto op    = GetKVClientOp();
  auto kv_op = std::dynamic_pointer_cast<KVClientOp>(op);
  if (kv_op) {
    kv_op->SetPSConfig(host, static_cast<int>(port));
  } else {
    LOG(ERROR) << "Failed to cast CommonOp to KVClientOp. Cannot set "
                  "PS config.";
    throw std::runtime_error(
        "Failed to set PS config: storage backend is not KVClientOp");
  }
}

TORCH_LIBRARY(recstore_ops, m) {
  m.def("emb_read", emb_read_torch);
  m.def("emb_update", emb_update_torch);
  m.def("emb_update_table", emb_update_table_torch);
  m.def("init_embedding_table", init_embedding_table_torch);
  m.def("emb_write", emb_write_torch);
  m.def("emb_prefetch", emb_prefetch_torch);
  m.def("emb_wait_result", emb_wait_result_torch);
  m.def("set_ps_config", set_ps_config_torch);
}

} // namespace framework
} // namespace recstore
