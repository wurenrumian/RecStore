#include "optimizer.h"
#include <cstring>

void SGD::Init(const std::vector<std::string> table_name,
               const EmbeddingTableConfig& config,
               BaseKV* base_kv) {
  for (const auto& name : table_name) {
    SparseTensor* param_tensor  = new SparseTensor();
    std::vector<uint64_t> shape = {config.num_embeddings, config.embedding_dim};
    TAG_TYPE tag                = 0; // PARAMETER tag
    param_tensor->init(
        const_cast<std::string&>(name), PARAMETER, tag, shape, base_kv);
    tensor_map_[name] = param_tensor;
  }
}

void SGD::Update(std::string table,
                 const std::vector<uint64_t>& keys,
                 const std::vector<std::vector<float>>& grads,
                 unsigned tid) {
  std::vector<std::vector<float>> current_values(keys.size());
  auto it = tensor_map_.find(table);

  if (it == tensor_map_.end()) {
    throw std::runtime_error("Table not found: " + table);
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    std::string value;
    it->second->Get(keys[i], value, tid);
    if (value.empty()) {
      current_values[i] = std::vector<float>(grads[i].size(), 0.0f);
    } else {
      size_t float_count = value.size() / sizeof(float);
      if (float_count != grads[i].size()) {
        current_values[i] = std::vector<float>(grads[i].size(), 0.0f);
      } else {
        float* data       = (float*)value.data();
        current_values[i] = std::vector<float>(data, data + grads[i].size());
      }
    }
  }

  std::vector<std::vector<float>> updated_values;
  updated_values.reserve(keys.size());

  for (size_t i = 0; i < keys.size(); ++i) {
    std::vector<float> updated(grads[i].size());
    for (size_t j = 0; j < grads[i].size(); ++j) {
      updated[j] = current_values[i][j] - learning_rate_ * grads[i][j];
    }
    updated_values.push_back(updated);
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    std::string value((char*)updated_values[i].data(),
                      updated_values[i].size() * sizeof(float));
    it->second->Put(keys[i], value, tid);
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
                     const std::vector<std::vector<float>>& grads,
                     unsigned tid) {
  std::vector<std::vector<float>> current_values(keys.size());
  std::vector<std::vector<float>> accumulated_gradients_squared(keys.size());

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
    std::string value;
    param_it->second->Get(keys[i], value, tid);
    if (value.empty()) {
      current_values[i] = std::vector<float>(grads[i].size(), 0.0f);
    } else {
      size_t float_count = value.size() / sizeof(float);
      if (float_count != grads[i].size()) {
        current_values[i] = std::vector<float>(grads[i].size(), 0.0f);
      } else {
        float* data       = (float*)value.data();
        current_values[i] = std::vector<float>(data, data + grads[i].size());
      }
    }

    acc_it->second->Get(keys[i], value, tid);
    if (value.empty()) {
      accumulated_gradients_squared[i] =
          std::vector<float>(grads[i].size(), 0.0f);
    } else {
      size_t float_count = value.size() / sizeof(float);
      if (float_count != grads[i].size()) {
        accumulated_gradients_squared[i] =
            std::vector<float>(grads[i].size(), 0.0f);
      } else {
        float* data = (float*)value.data();
        accumulated_gradients_squared[i] =
            std::vector<float>(data, data + grads[i].size());
      }
    }
  }

  std::vector<std::vector<float>> updated_values;
  std::vector<std::vector<float>> updated_accumulated_gradients;

  updated_values.reserve(keys.size());
  updated_accumulated_gradients.reserve(keys.size());

  for (size_t i = 0; i < keys.size(); ++i) {
    std::vector<float> updated(grads[i].size());
    std::vector<float> updated_acc_grad(grads[i].size());

    for (size_t j = 0; j < grads[i].size(); ++j) {
      updated_acc_grad[j] =
          accumulated_gradients_squared[i][j] + grads[i][j] * grads[i][j];
      float adaptive_lr =
          learning_rate_ / (std::sqrt(updated_acc_grad[j]) + epsilon_);
      updated[j] = current_values[i][j] - adaptive_lr * grads[i][j];
    }

    updated_values.push_back(updated);
    updated_accumulated_gradients.push_back(updated_acc_grad);
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    std::string param_value((char*)updated_values[i].data(),
                            updated_values[i].size() * sizeof(float));
    std::string acc_value(
        (char*)updated_accumulated_gradients[i].data(),
        updated_accumulated_gradients[i].size() * sizeof(float));

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
                            const std::vector<std::vector<float>>& grads,
                            unsigned tid) {
  std::vector<std::vector<float>> current_values(keys.size());
  std::vector<std::vector<float>> accumulated_gradients_squared(keys.size());

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
    std::string value;
    param_it->second->Get(keys[i], value, tid);
    if (value.empty()) {
      current_values[i] = std::vector<float>(grads[i].size(), 0.0f);
    } else {
      size_t float_count = value.size() / sizeof(float);
      if (float_count != grads[i].size()) {
        current_values[i] = std::vector<float>(grads[i].size(), 0.0f);
      } else {
        float* data       = (float*)value.data();
        current_values[i] = std::vector<float>(data, data + grads[i].size());
      }
    }

    acc_it->second->Get(keys[i], value, tid);
    if (value.empty()) {
      accumulated_gradients_squared[i] = std::vector<float>(1, 0.0f);
    } else {
      size_t float_count = value.size() / sizeof(float);
      if (float_count < 1) {
        accumulated_gradients_squared[i] = std::vector<float>(1, 0.0f);
      } else {
        float* data                      = (float*)value.data();
        accumulated_gradients_squared[i] = std::vector<float>(data, data + 1);
      }
    }
  }

  std::vector<std::vector<float>> updated_values;
  std::vector<std::vector<float>> updated_accumulated_gradients;

  updated_values.reserve(keys.size());
  updated_accumulated_gradients.reserve(keys.size());

  for (size_t i = 0; i < keys.size(); ++i) {
    std::vector<float> updated(grads[i].size());
    std::vector<float> updated_acc_grad(1);

    float grad_square_mean = 0.0;
    for (size_t j = 0; j < grads[i].size(); ++j) {
      grad_square_mean += grads[i][j] * grads[i][j];
    }
    grad_square_mean /= grads[i].size();

    updated_acc_grad[0] =
        accumulated_gradients_squared[i][0] + grad_square_mean;

    float adaptive_lr =
        learning_rate_ / (std::sqrt(updated_acc_grad[0]) + epsilon_);
    for (size_t j = 0; j < grads[i].size(); ++j) {
      updated[j] = current_values[i][j] - adaptive_lr * grads[i][j];
    }

    updated_values.push_back(updated);
    updated_accumulated_gradients.push_back(updated_acc_grad);
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    std::string param_value((char*)updated_values[i].data(),
                            updated_values[i].size() * sizeof(float));
    std::string acc_value(
        (char*)updated_accumulated_gradients[i].data(),
        updated_accumulated_gradients[i].size() * sizeof(float));

    param_it->second->Put(keys[i], param_value, tid);
    acc_it->second->Put(keys[i], acc_value, tid);
  }
}
