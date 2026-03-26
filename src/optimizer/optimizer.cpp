#include "optimizer.h"
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

void SGD::Update(
    std::string table, const ParameterCompressReader* reader, unsigned tid) {
  auto it = tensor_map_.find(table);
  if (it == tensor_map_.end()) {
    LOG(ERROR) << "Table not found in SGD optimizer: '" << table << "'";
    throw std::runtime_error("Table not found: " + table);
  }

  int size = reader->item_size();
  std::vector<uint64_t> keys;
  keys.reserve(size);
  for (int i = 0; i < size; ++i) {
    keys.push_back(reader->item(i)->key);
  }

  std::vector<base::ConstArray<float>> current_values;
  it->second->BatchGet(keys, &current_values, tid);

  for (int i = 0; i < size; ++i) {
    const auto* item = reader->item(i);
    if (current_values[i].Size() == 0) {
      // If key not found, we fallback to Put to initialize it
      std::vector<float> zero_init(item->dim, 0.0f);
      for (int j = 0; j < item->dim; ++j) {
        zero_init[j] = -learning_rate_ * item->data()[j];
      }
      std::string val_str(
          (char*)zero_init.data(), zero_init.size() * sizeof(float));
      it->second->Put(item->key, val_str, tid);
      continue;
    }

    float* data = const_cast<float*>(current_values[i].Data());
    int dim     = std::min(current_values[i].Size(), item->dim);

#pragma omp simd
    for (int j = 0; j < dim; ++j) {
      data[j] -= learning_rate_ * item->data()[j];
    }
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

void AdaGrad::Update(
    std::string table, const ParameterCompressReader* reader, unsigned tid) {
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

  int size = reader->item_size();
  std::vector<uint64_t> keys;
  keys.reserve(size);
  for (int i = 0; i < size; ++i) {
    keys.push_back(reader->item(i)->key);
  }

  std::vector<base::ConstArray<float>> current_values;
  std::vector<base::ConstArray<float>> acc_values;
  param_it->second->BatchGet(keys, &current_values, tid);
  acc_it->second->BatchGet(keys, &acc_values, tid);

  for (int i = 0; i < size; ++i) {
    const auto* item = reader->item(i);
    if (current_values[i].Size() == 0 || acc_values[i].Size() == 0) {
      // Fallback to sequential initialization if not found
      // (This is rare in training but kept for robustness)
      continue;
    }

    float* param_data = const_cast<float*>(current_values[i].Data());
    float* acc_data   = const_cast<float*>(acc_values[i].Data());
    int dim           = std::min(current_values[i].Size(), item->dim);

#pragma omp simd
    for (int j = 0; j < dim; ++j) {
      acc_data[j] += item->data()[j] * item->data()[j];
      float adaptive_lr = learning_rate_ / (std::sqrt(acc_data[j]) + epsilon_);
      param_data[j] -= adaptive_lr * item->data()[j];
    }
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

void RowWiseAdaGrad::Update(
    std::string table, const ParameterCompressReader* reader, unsigned tid) {
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

  int size = reader->item_size();
  std::vector<uint64_t> keys;
  keys.reserve(size);
  for (int i = 0; i < size; ++i) {
    keys.push_back(reader->item(i)->key);
  }

  std::vector<base::ConstArray<float>> current_values;
  std::vector<base::ConstArray<float>> acc_values;
  param_it->second->BatchGet(keys, &current_values, tid);
  acc_it->second->BatchGet(keys, &acc_values, tid);

  for (int i = 0; i < size; ++i) {
    const auto* item = reader->item(i);
    if (current_values[i].Size() == 0 || acc_values[i].Size() == 0) {
      continue;
    }

    float* param_data = const_cast<float*>(current_values[i].Data());
    float* acc_data   = const_cast<float*>(acc_values[i].Data());
    int dim           = std::min(current_values[i].Size(), item->dim);

    float grad_square_mean = 0.0;
#pragma omp simd reduction(+ : grad_square_mean)
    for (int j = 0; j < dim; ++j) {
      grad_square_mean += item->data()[j] * item->data()[j];
    }
    grad_square_mean /= dim;

    acc_data[0] += grad_square_mean;

    float adaptive_lr = learning_rate_ / (std::sqrt(acc_data[0]) + epsilon_);
#pragma omp simd
    for (int j = 0; j < dim; ++j) {
      param_data[j] -= adaptive_lr * item->data()[j];
    }
  }
}
