#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>
#include <stdlib.h>

typedef struct {
    float info;
    float grad;
    void (*_backward)(); // A function to propagate gradients
    struct Value *prev[2]; // Keep track of input values for backward pass
} Value;

// Structure to represent storage for tensor data
typedef struct {
    Value *data;
} Storage;

// Structure to represent a view of the tensor
typedef struct {
    Storage *storage;  // Pointer to the storage of tensor data
    int stride;        // Stride of the tensor
    int shape;         // Shape of the tensor
    int offset;         // Offset of the tensor
} View;

// Structure to represent a 1D tensor
typedef struct {
    int id;      // Unique identifier for the tensor
    View *view;  // Pointer to the tensor's view

    bool requires_grad;
} Tensor1d;

Tensor1d* init_tensor(int size, int stride);
Tensor1d* tensor_arange(int upper_limit);
bool verify_tensor(Tensor1d *tensor);
void append_data(Tensor1d *tensor, int new_value);
Tensor1d* add_tensor_to_tensor(Tensor1d *tensor, Tensor1d *tensor_);
Tensor1d* mul_tensor_to_tensor(Tensor1d *tensor, Tensor1d *tensor_);
void set_item(Tensor1d *tensor, float item, int index);
int get_size(Tensor1d *tensor);
Tensor1d* get_item_as_tensor(Tensor1d *tensor, int index);
float get_item(Tensor1d *tensor, int index);
Tensor1d* get_slice(Tensor1d *tensor, int start, int end, int stride);
void free_tensor(Tensor1d *tensor);
int min(int a, int b);
int max(int a, int b);
Tensor1d* add_tensor_broadcast(Tensor1d *tensor, Tensor1d *small_tensor);
Tensor1d* tensor_scalar_add(Tensor1d *tensor, float scalar);
Tensor1d* tensor_scalar_mul(Tensor1d *tensor, float scalar);
Tensor1d* get_slice_default(Tensor1d *tensor, int start, int end, int stride);
void print_tensor(Tensor1d *tensor);
char* tensor_to_string(Tensor1d *tensor);

void backward_add(Tensor1d *tensor, Tensor1d *tensor_, int i, Tensor1d *result);
void backward_mul(Tensor1d *tensor, Tensor1d *tensor_, int i, Tensor1d *result);
void backward(Tensor1d *tensor, Tensor1d *tensor_, Tensor1d *result);
void zero_grad(Tensor1d *tensor);
Tensor1d* tensor_sum(Tensor1d *tensor);
void add_grad(Tensor1d *tensor, int index, float value);
Tensor1d* get_grad(Tensor1d *tensor);
#endif // TENSOR_H
