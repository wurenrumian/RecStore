#include "optimizer.h"
#include <algorithm>
#include <cstring>

void SGD::Init(const std::vector<std::string> table_name,
               const EmbeddingTableConfig& config,
               BaseKV* base_kv) {
  LOG(INFO) << "SGD::Init called with " << table_name.size() << " table(s)";
  for (const auto& name : table_name) {
    LOG(INFO) << "  Initializing table: '" << name << "' with shape ["
              << config.num_embeddings << ", " << config.embedding_dim << "]";
    SparseTensor* param_tensor  = new SparseTensor();
    std::vector<uint64_t> shape = {config.num_embeddings, config.embedding_dim};
    TAG_TYPE tag                = 0; // PARAMETER tag
    param_tensor->init(
        const_cast<std::string&>(name), PARAMETER, tag, shape, base_kv);
    tensor_map_[name] = param_tensor;
  }
  LOG(INFO) << "SGD::Init completed. tensor_map_ now has " << tensor_map_.size()
            << " entries";
}

void SGD::Update(std::string table,
                 const std::vector<uint64_t>& keys,
                 const ParameterCompressReader* grads,
                 unsigned tid) {
  if (grads == nullptr) {
    throw std::runtime_error("SGD gradients reader is null");
  }
  if (keys.size() != static_cast<size_t>(grads->item_size())) {
    throw std::runtime_error("SGD keys/grads size mismatch");
  }

  auto it = tensor_map_.find(table);

  if (it == tensor_map_.end()) {
    LOG(ERROR) << "Table not found in SGD optimizer: '" << table << "'";
    LOG(ERROR) << "Available tables (" << tensor_map_.size() << "):";
    for (const auto& pair : tensor_map_) {
      LOG(ERROR) << "  - '" << pair.first << "'";
    }
    throw std::runtime_error("Table not found: " + table);
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    const auto* grad_item = grads->item(i);
    const int grad_dim    = grad_item->dim;
    const float* grad_ptr = grad_item->data();

    std::string value;
    it->second->Get(keys[i], value, tid);
    std::vector<float> current_value(grad_dim, 0.0f);
    if (value.empty()) {
    } else {
      size_t float_count = value.size() / sizeof(float);
      if (float_count != static_cast<size_t>(grad_dim)) {
      } else {
        const float* data = reinterpret_cast<const float*>(value.data());
        std::copy_n(data, grad_dim, current_value.data());
      }
    }

    std::vector<float> updated(grad_dim);
    for (int j = 0; j < grad_dim; ++j) {
      updated[j] = current_value[j] - learning_rate_ * grad_ptr[j];
    }

    std::string updated_value(reinterpret_cast<const char*>(updated.data()),
                              updated.size() * sizeof(float));
    it->second->Put(keys[i], updated_value, tid);
  }
}

void AdaGrad::Init(const std::vector<std::string> table_name,
                   const EmbeddingTableConfig& config,
                   BaseKV* base_kv) {
  for (const auto& name : table_name) {
    SparseTensor* param_tensor  = new SparseTensor();
    std::vector<uint64_t> shape = {config.num_embeddings, config.embedding_dim};
    TAG_TYPE tag                = 0;
    param_tensor->init(
        const_cast<std::string&>(name), PARAMETER, tag, shape, base_kv);
    tensor_map_[name] = param_tensor;

    std::string acc_table_name = name + "_accumulated_grad";
    SparseTensor* acc_tensor   = new SparseTensor();
    acc_tensor->init(
        const_cast<std::string&>(acc_table_name),
        MOMENT_1,
        tag,
        shape,
        base_kv);
    tensor_map_[acc_table_name] = acc_tensor;
  }
}

void AdaGrad::Update(std::string table,
                     const std::vector<uint64_t>& keys,
                     const ParameterCompressReader* grads,
                     unsigned tid) {
  if (grads == nullptr) {
    throw std::runtime_error("AdaGrad gradients reader is null");
  }
  if (keys.size() != static_cast<size_t>(grads->item_size())) {
    throw std::runtime_error("AdaGrad keys/grads size mismatch");
  }

  auto param_it = tensor_map_.find(table);
  if (param_it == tensor_map_.end()) {
    throw std::runtime_error("Table not found: " + table);
  }

  std::string acc_table = table + "_accumulated_grad";
  auto acc_it           = tensor_map_.find(acc_table);
  if (acc_it == tensor_map_.end()) {
    throw std::runtime_error(
        "Accumulated gradient table not found: " + acc_table);
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    const auto* grad_item = grads->item(i);
    const int grad_dim    = grad_item->dim;
    const float* grad_ptr = grad_item->data();

    std::string value;
    param_it->second->Get(keys[i], value, tid);
    std::vector<float> current_value(grad_dim, 0.0f);
    if (value.empty()) {
    } else {
      size_t float_count = value.size() / sizeof(float);
      if (float_count != static_cast<size_t>(grad_dim)) {
      } else {
        const float* data = reinterpret_cast<const float*>(value.data());
        std::copy_n(data, grad_dim, current_value.data());
      }
    }

    acc_it->second->Get(keys[i], value, tid);
    std::vector<float> accumulated_gradient_squared(grad_dim, 0.0f);
    if (value.empty()) {
    } else {
      size_t float_count = value.size() / sizeof(float);
      if (float_count != static_cast<size_t>(grad_dim)) {
      } else {
        const float* data = reinterpret_cast<const float*>(value.data());
        std::copy_n(data, grad_dim, accumulated_gradient_squared.data());
      }
    }

    std::vector<float> updated(grad_dim);
    std::vector<float> updated_acc_grad(grad_dim);

    for (int j = 0; j < grad_dim; ++j) {
      updated_acc_grad[j] =
          accumulated_gradient_squared[j] + grad_ptr[j] * grad_ptr[j];
      float adaptive_lr =
          learning_rate_ / (std::sqrt(updated_acc_grad[j]) + epsilon_);
      updated[j] = current_value[j] - adaptive_lr * grad_ptr[j];
    }

    std::string param_value(reinterpret_cast<const char*>(updated.data()),
                            updated.size() * sizeof(float));
    std::string acc_value(reinterpret_cast<const char*>(updated_acc_grad.data()),
                          updated_acc_grad.size() * sizeof(float));

    param_it->second->Put(keys[i], param_value, tid);
    acc_it->second->Put(keys[i], acc_value, tid);
  }
}

void RowWiseAdaGrad::Init(const std::vector<std::string> table_name,
                          const EmbeddingTableConfig& config,
                          BaseKV* base_kv) {
  for (const auto& name : table_name) {
    SparseTensor* param_tensor  = new SparseTensor();
    std::vector<uint64_t> shape = {config.num_embeddings, config.embedding_dim};
    TAG_TYPE tag                = 0;
    param_tensor->init(
        const_cast<std::string&>(name), PARAMETER, tag, shape, base_kv);
    tensor_map_[name] = param_tensor;

    std::string acc_table_name      = name + "_rowwise_accumulated_grad";
    SparseTensor* acc_tensor        = new SparseTensor();
    std::vector<uint64_t> acc_shape = {
        config.num_embeddings, 1}; // One value per row
    acc_tensor->init(
        const_cast<std::string&>(acc_table_name),
        MOMENT_1,
        tag,
        acc_shape,
        base_kv);
    tensor_map_[acc_table_name] = acc_tensor;
  }
}

void RowWiseAdaGrad::Update(std::string table,
                            const std::vector<uint64_t>& keys,
                            const ParameterCompressReader* grads,
                            unsigned tid) {
  if (grads == nullptr) {
    throw std::runtime_error("RowWiseAdaGrad gradients reader is null");
  }
  if (keys.size() != static_cast<size_t>(grads->item_size())) {
    throw std::runtime_error("RowWiseAdaGrad keys/grads size mismatch");
  }

  auto param_it = tensor_map_.find(table);
  if (param_it == tensor_map_.end()) {
    throw std::runtime_error("Table not found: " + table);
  }

  std::string acc_table = table + "_rowwise_accumulated_grad";
  auto acc_it           = tensor_map_.find(acc_table);
  if (acc_it == tensor_map_.end()) {
    throw std::runtime_error(
        "Row-wise accumulated gradient table not found: " + acc_table);
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    const auto* grad_item = grads->item(i);
    const int grad_dim    = grad_item->dim;
    const float* grad_ptr = grad_item->data();

    std::string value;
    param_it->second->Get(keys[i], value, tid);
    std::vector<float> current_value(grad_dim, 0.0f);
    if (value.empty()) {
    } else {
      size_t float_count = value.size() / sizeof(float);
      if (float_count != static_cast<size_t>(grad_dim)) {
      } else {
        const float* data = reinterpret_cast<const float*>(value.data());
        std::copy_n(data, grad_dim, current_value.data());
      }
    }

    acc_it->second->Get(keys[i], value, tid);
    float accumulated_gradient_squared = 0.0f;
    if (value.empty()) {
    } else {
      size_t float_count = value.size() / sizeof(float);
      if (float_count < 1) {
      } else {
        const float* data         = reinterpret_cast<const float*>(value.data());
        accumulated_gradient_squared = data[0];
      }
    }

    std::vector<float> updated(grad_dim);
    std::vector<float> updated_acc_grad(1);

    float grad_square_mean = 0.0;
    for (int j = 0; j < grad_dim; ++j) {
      grad_square_mean += grad_ptr[j] * grad_ptr[j];
    }
    grad_square_mean /= grad_dim;

    updated_acc_grad[0] = accumulated_gradient_squared + grad_square_mean;

    float adaptive_lr =
        learning_rate_ / (std::sqrt(updated_acc_grad[0]) + epsilon_);
    for (int j = 0; j < grad_dim; ++j) {
      updated[j] = current_value[j] - adaptive_lr * grad_ptr[j];
    }

    std::string param_value(reinterpret_cast<const char*>(updated.data()),
                            updated.size() * sizeof(float));
    std::string acc_value(reinterpret_cast<const char*>(updated_acc_grad.data()),
                          updated_acc_grad.size() * sizeof(float));

    param_it->second->Put(keys[i], param_value, tid);
    acc_it->second->Put(keys[i], acc_value, tid);
  }
}
