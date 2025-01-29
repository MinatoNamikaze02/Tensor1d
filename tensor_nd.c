#include <stdio.h>
#include <math.h>
#include <string.h> 
#include "tensor_nd.h"

int min(int a, int b) {
    return (a < b) ? a : b;
}

int max(int a, int b) {
    return (a > b) ? a : b;
}

Tensor1d* init_tensor(int size, int stride) {
    Tensor1d *tensor = (Tensor1d *)malloc(sizeof(Tensor1d));
    if (!tensor) {
        fprintf(stderr, "Error: Memory allocation for Tensor1d failed.\n");
        return NULL;
    }

    tensor->view = (View *)malloc(sizeof(View));
    if (!tensor->view) {
        fprintf(stderr, "Error: Memory allocation for View failed.\n");
        free(tensor);
        return NULL;
    }

    tensor->view->storage = (Storage *)malloc(sizeof(Storage));
    if (!tensor->view->storage) {
        fprintf(stderr, "Error: Memory allocation for Storage failed.\n");
        free(tensor->view);
        free(tensor);
        return NULL;
    }

    tensor->view->storage->data = (Value *)malloc(size * sizeof(Value));
    if (!tensor->view->storage->data) {
        fprintf(stderr, "Error: Memory allocation for data failed.\n");
        free(tensor->view->storage);
        free(tensor->view);
        free(tensor);
        return NULL;
    }

    tensor->id = 1;
    tensor->view->shape = size;
    tensor->view->stride = stride;

    for (int i = 0; i < size; i++) {
        tensor->view->storage->data[i].grad = 0.0f;
        tensor->view->storage->data[i]._backward = NULL;
        tensor->view->storage->data[i].prev[0] = NULL;
        tensor->view->storage->data[i].prev[1] = NULL;
    }

    return tensor;
}

Tensor1d* tensor_arange(int upper_limit) {
    Tensor1d *tensor = init_tensor(upper_limit, 1);
    if (!tensor) {
        return NULL;
    }

    for (int i = 0; i < upper_limit; i++) {
        tensor->view->storage->data[i].info = i;
        tensor->view->storage->data[i].grad = 0;
    }

    return tensor;
}

Tensor1d* add_tensor_broadcast(Tensor1d *tensor, Tensor1d *small_tensor) {
    if (!verify_tensor(tensor) || !verify_tensor(small_tensor)) {
        return NULL;
    }

    int larger_size = tensor->view->shape;
    int smaller_size = small_tensor->view->shape;

    if (larger_size % smaller_size != 0) {
        fprintf(stderr, "Error: Broadcast size mismatch.\n");
        return NULL;
    }

    Tensor1d *result = init_tensor(larger_size, 1);
    if (!result) {
        return NULL;
    }

    for (int i = 0; i < larger_size; i++) {
        result->view->storage->data[i].info = tensor->view->storage->data[i].info +
                                               small_tensor->view->storage->data[i % smaller_size].info;
        result->view->storage->data[i].grad = 0;
    }

    return result;
}

Tensor1d* tensor_scalar_add(Tensor1d *tensor, float scalar) {
    if (!verify_tensor(tensor)) {
        return NULL;
    }

    Tensor1d *result = init_tensor(tensor->view->shape, 1);
    if (!result) {
        return NULL;
    }

    for (int i = 0; i < tensor->view->shape; i++) {
        result->view->storage->data[i].info = tensor->view->storage->data[i].info + scalar;
        result->view->storage->data[i].grad = 0;
    }
    return result;
}

Tensor1d* tensor_scalar_mul(Tensor1d *tensor, float scalar) {
    if (!verify_tensor(tensor)) {
        return NULL;
    }

    Tensor1d *result = init_tensor(tensor->view->shape, 1);
    if (!result) {
        return NULL;
    }

    for (int i = 0; i < tensor->view->shape; i++) {
        result->view->storage->data[i].info = tensor->view->storage->data[i].info * scalar;
        result->view->storage->data[i].grad = 0;
    }
    return result;
}

void print_tensor(Tensor1d *tensor) {
    if (!verify_tensor(tensor)) {
        printf("Invalid tensor.\n");
        return;
    }

    printf("Tensor (size: %d): [", tensor->view->shape);
    for (int i = 0; i < tensor->view->shape; i++) {
        printf("%.2f", tensor->view->storage->data[i].info);
        if (i < tensor->view->shape - 1) printf(", ");
    }
    printf("]\n");
}

bool verify_tensor(Tensor1d *tensor) {
    if (!tensor || !tensor->view || !tensor->view->storage || !tensor->view->storage->data) {
        return 0;
    }
    return 1;
}

void append_data(Tensor1d *tensor, int new_value) {
    if (!verify_tensor(tensor)) {
        return;
    }
    int new_size = tensor->view->shape + 1;
    Value *new_data = (Value *)realloc(tensor->view->storage->data, new_size * sizeof(Value));

    if (!new_data) {
        fprintf(stderr, "Error: Memory reallocation failed.\n");
        return;
    }

    tensor->view->storage->data = new_data;
    tensor->view->storage->data[tensor->view->shape].info = new_value;
    tensor->view->storage->data[tensor->view->shape].grad = 0;
    tensor->view->shape = new_size;
}

void backward_add(Tensor1d *tensor, Tensor1d *tensor_, int i, Tensor1d *result) {
    tensor->view->storage->data[i].grad += result->view->storage->data[i].grad;
    tensor_->view->storage->data[i].grad += result->view->storage->data[i].grad;
}

void backward_mul(Tensor1d *tensor, Tensor1d *tensor_, int i, Tensor1d *result) {
    tensor->view->storage->data[i].grad += tensor_->view->storage->data[i].info * result->view->storage->data[i].grad;
    tensor_->view->storage->data[i].grad += tensor->view->storage->data[i].info * result->view->storage->data[i].grad;
}

Tensor1d* get_grad(Tensor1d *tensor) {
    if (!verify_tensor(tensor)) {
        return NULL;
    }
    Tensor1d *grad = init_tensor(tensor->view->shape, 1);
    if (!grad) {
        return NULL;
    }
    for (int i = 0; i < tensor->view->shape; i++) {
        grad->view->storage->data[i].info = tensor->view->storage->data[i].grad;
    }
    return grad;
}

void backward(Tensor1d *tensor, Tensor1d *tensor_, Tensor1d *result) {
    for (int i = 0; i < tensor->view->shape; i++) {
        if (tensor->view->storage->data[i]._backward) {
            tensor->view->storage->data[i]._backward(tensor, tensor_, i, result);
        }
    }
}

void zero_grad(Tensor1d *tensor) {
    if (!verify_tensor(tensor)) {
        return;
    }
    for (int i = 0; i < tensor->view->shape; i++) {
        tensor->view->storage->data[i].grad = 0.0f;
    }
}

void add_grad(Tensor1d *tensor, int index, float grad) {
    if (verify_tensor(tensor) && index < tensor->view->shape) {
        tensor->view->storage->data[index].grad += grad;
    }
}

Tensor1d* add_tensor_to_tensor(Tensor1d *tensor, Tensor1d *tensor_) {
    if (!verify_tensor(tensor) || !verify_tensor(tensor_)) {
        return NULL;
    }
    if (tensor->view->shape != tensor_->view->shape) {
        fprintf(stderr, "Error: Tensor shapes do not match.\n");
        return NULL;
    }
    Tensor1d *result = init_tensor(tensor->view->shape, 1);
    if (!result) {
        return NULL;
    }
    for (int i = 0; i < tensor->view->shape; i++) {
        result->view->storage->data[i].info = tensor->view->storage->data[i].info + tensor_->view->storage->data[i].info;
        result->view->storage->data[i].grad = 0;
        result->view->storage->data[i]._backward = backward_add;
    }
    return result;
}

Tensor1d* mul_tensor_to_tensor(Tensor1d *tensor, Tensor1d *tensor_) {
    if (!verify_tensor(tensor) || !verify_tensor(tensor_)) {
        return NULL;
    }
    if (tensor->view->shape != tensor_->view->shape) {
        fprintf(stderr, "Error: Tensor shapes do not match.\n");
        return NULL;
    }
    Tensor1d *result = init_tensor(tensor->view->shape, 1);
    if (!result) {
        return NULL;
    }
    for (int i = 0; i < tensor->view->shape; i++) {
        result->view->storage->data[i].info = tensor->view->storage->data[i].info * tensor_->view->storage->data[i].info;
        result->view->storage->data[i].grad = 0;

        result->view->storage->data[i]._backward = backward_mul;
    }
    return result;
}

Tensor1d* tensor_sum(Tensor1d *tensor) {
    if (!verify_tensor(tensor)) {
        return NULL;
    }
    float sum = 0;
    for (int i = 0; i < tensor->view->shape; i++) {
        sum += tensor->view->storage->data[i].info;
    }
    Tensor1d *result = init_tensor(1, 1);
    if (!result) {
        return NULL;
    }
    result->view->storage->data[0].info = sum;
    result->view->storage->data[0].grad = 0;
    return result;
}

void set_item(Tensor1d *tensor, float item, int index) {
    // support negative index slicing
    if (index < 0) { index = tensor->view->shape + index; }

    // handle out of bounds values?
    index = min(max(index, 0), tensor->view->shape);

    if (!verify_tensor(tensor)) {
        return;
    }
    tensor->view->storage->data[index].info = item;
    tensor->view->storage->data[index].grad = 0;
}

int get_size(Tensor1d *tensor) {
    if (!verify_tensor(tensor)) {
        return -1;
    }
    return tensor->view->shape;
}

void get_tensor_data(Tensor1d *tensor, float *buffer, int size) {
    if (!verify_tensor(tensor) || size < tensor->view->shape) return;
    
    for (int i = 0; i < tensor->view->shape; i++) {
        buffer[i] = tensor->view->storage->data[i].info;
    }
}

float get_item(Tensor1d *tensor, int index){
    if (!verify_tensor(tensor)) {
        return -1;
    }
    return tensor->view->storage->data[index].info;
}

Tensor1d* get_slice(Tensor1d *tensor, int start, int end, int stride) {
    if (!verify_tensor(tensor)) {
        fprintf(stderr, "Error: Invalid tensor.\n");
        return NULL;
    }

    int shape = tensor->view->shape;
    if (start < 0) start += shape;
    if (end < 0) end += shape;

    start = start < 0 ? 0 : (start >= shape ? shape : start);
    end = end < 0 ? 0 : (end >= shape ? shape : end);

    if (stride <= 0) {
        fprintf(stderr, "Error: Stride must be positive.\n");
        return NULL;
    }
    int size = (end - start + stride - 1) / stride;
    if (start >= end || size <= 0) {
        fprintf(stderr, "Error: Invalid slice parameters (start >= end or size <= 0).\n");
        return NULL;
    }

    Tensor1d *slice_tensor = init_tensor(size, stride);
    if (!slice_tensor) {
        fprintf(stderr, "Error: Memory allocation for slice tensor failed.\n");
        return NULL;
    }
    int idx = 0;
    for (int i = start; i < end; i += stride) {
        slice_tensor->view->storage->data[idx++] = tensor->view->storage->data[i];
        slice_tensor->view->storage->data[idx].grad = tensor->view->storage->data[i].grad;
    }

    return slice_tensor;
}

Tensor1d* get_item_as_tensor(Tensor1d *tensor, int index) {
    if (!verify_tensor(tensor)) {
        return NULL;
    }
    float value = tensor->view->storage->data[index].info;
    Tensor1d *tensor_ = init_tensor(1, 1);

    if (!tensor_) {
        fprintf(stderr, "Error: Memory allocation for new tensor failed.\n");
        return NULL;
    }

    tensor_->view->storage->data[0].info = value;
    tensor_->view->storage->data[0].grad = 0;
    return tensor_;
}

char* tensor_to_string(Tensor1d *tensor) {
    if (!verify_tensor(tensor)) {
        return NULL;
    }
    char *str = (char *)malloc(tensor->view->shape * 10 + 20);
    if (!str) {
        return NULL;
    }
    sprintf(str, "Tensor (size: %d): [", tensor->view->shape);
    for (int i = 0; i < tensor->view->shape; i++) {
        char buf[10];
        sprintf(buf, "%.2f", tensor->view->storage->data[i].info);
        strcat(str, buf);
        if (i < tensor->view->shape - 1) {
            strcat(str, ", ");
        }
    }
    strcat(str, "]");
    return str;
}

void free_tensor(Tensor1d *tensor) {
    if (tensor) {
        if (tensor->view && tensor->view->storage && tensor->view->storage->data) {
            free(tensor->view->storage->data);
        }
        free(tensor->view->storage);
        free(tensor->view);
        free(tensor);
    }
}
