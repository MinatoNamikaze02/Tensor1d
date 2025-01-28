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

    tensor->view->storage->data = (float *)malloc(size * sizeof(float));
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

    return tensor;
}

Tensor1d* tensor_arange(int upper_limit) {
    Tensor1d *tensor = init_tensor(upper_limit, 1);
    if (!tensor) {
        return NULL;
    }

    for (int i = 0; i < upper_limit; i++) {
        tensor->view->storage->data[i] = i;
    }

    return tensor;
}


Tensor1d* add_tensor_broadcast(Tensor1d *tensor, Tensor1d *small_tensor) {
    if (!verify_tensor(tensor) || !verify_tensor(small_tensor)) {
        return NULL;
    }

    int larger_size = tensor->view->shape;
    int smaller_size = small_tensor->view->shape;

    // Ensure the smaller tensor can broadcast
    if (larger_size % smaller_size != 0) {
        fprintf(stderr, "Error: Broadcast size mismatch.\n");
        return NULL;
    }

    Tensor1d *result = init_tensor(larger_size, 1);
    if (!result) {
        return NULL;
    }

    for (int i = 0; i < larger_size; i++) {
        result->view->storage->data[i] = tensor->view->storage->data[i] +
                                         small_tensor->view->storage->data[i % smaller_size];
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
        result->view->storage->data[i] = tensor->view->storage->data[i] + scalar;
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
        printf("%.2f", tensor->view->storage->data[i]);
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

void append_data(Tensor1d *tensor, int new_value){
    if (!verify_tensor(tensor)) {
        return;
    }
    int new_size = tensor->view->shape + 1;
    float *new_data = (float *)realloc(tensor->view->storage->data, new_size * sizeof(float));

    if (!new_data) {
        fprintf(stderr, "Error: Memory reallocation failed.\n");
        return;
    }

    tensor->view->storage->data = new_data;
    tensor->view->storage->data[tensor->view->shape] = new_value;
    tensor->view->shape = new_size;
}

Tensor1d* add_tensor_to_tensor(Tensor1d *tensor, Tensor1d *tensor_){
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
        result->view->storage->data[i] = tensor->view->storage->data[i] + tensor_->view->storage->data[i];
    }
    return result;
}

void set_item(Tensor1d *tensor, float item, int index){
    // suppport negative index slicing
    if (index < 0) { index = tensor->view->shape + index; }

    // hamdle out of bounds values?
    index = min(max(index, 0), tensor->view->shape);

    if (!verify_tensor(tensor)) {
        // printf("Invalid tensor.\n");
        return;
    }
    tensor->view->storage->data[index] = item;
}

int get_size(Tensor1d *tensor){
    if (!verify_tensor(tensor)) {
        return -1;
    }
    return tensor->view->shape;
}

Tensor1d* get_item_as_tensor(Tensor1d *tensor, int index){
    if (!verify_tensor(tensor)) {
        return NULL;
    }
    float value = tensor->view->storage->data[index];
    Tensor1d *tensor_ = init_tensor(1, 1);
    
    if (!tensor_) {
        fprintf(stderr, "Error: Memory allocation for new tensor failed.\n");
        return NULL;
    }

    tensor_->view->storage->data[0] = value;
    return tensor_;
}

char* tensor_to_string(Tensor1d *tensor) {
    if (!verify_tensor(tensor)) {
        return NULL;
    }
    
    int buf_size = 50 + tensor->view->shape * 15;
    char *str = (char *)malloc(buf_size * sizeof(char));
    
    char *ptr = str;
    ptr += sprintf(ptr, "Tensor (size: %d): [", tensor->view->shape);
    for (int i = 0; i < tensor->view->shape; i++) {
        if (i > 0) ptr += sprintf(ptr, ", ");
        ptr += sprintf(ptr, "%.2f", tensor->view->storage->data[i]);
    }
    sprintf(ptr, "]");
    return str;
}

void get_tensor_data(Tensor1d *tensor, float *buffer, int size) {
    if (!verify_tensor(tensor) || size < tensor->view->shape) return;
    
    for (int i = 0; i < tensor->view->shape; i++) {
        buffer[i] = tensor->view->storage->data[i];
    }
}


float get_item(Tensor1d *tensor, int index){
    if (!verify_tensor(tensor)) {
        return -1;
    }
    return tensor->view->storage->data[index];
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
    }

    return slice_tensor;
}

void free_tensor(Tensor1d *tensor) {
    if (!verify_tensor(tensor)) {
        return;
    }

    if (tensor->view) {
        if (tensor->view->storage) {
            free(tensor->view->storage->data);
            free(tensor->view->storage);
        }
        free(tensor->view);
    }
    free(tensor);
}

int main() {
    Tensor1d *tensor = tensor_arange(10);
    print_tensor(tensor);

    Tensor1d *tensor_ = tensor_arange(5);
    print_tensor(tensor_);

    Tensor1d *result = add_tensor_broadcast(tensor, tensor_);
    print_tensor(result);

    Tensor1d *result_ = tensor_scalar_add(tensor, 5);
    print_tensor(result_);

    Tensor1d *slice = get_slice(tensor, 2, 8, 2);
    print_tensor(slice);

    char *str = tensor_to_string(tensor);
    printf("%s\n", str);
    free(str);

    free_tensor(tensor);
    free_tensor(tensor_);
    free_tensor(result);
    free_tensor(result_);
    free_tensor(slice);

    return 0;
}