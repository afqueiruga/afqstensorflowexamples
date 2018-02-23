#include "tensorflow/c/c_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>

// Citation
// DrBBQ
// Jun 16 '17 at 11:31
// 
// https://stackoverflow.com/questions/44305647/segmentation-fault-when-using-tf-sessionrun-to-run-tensorflow-graph-in-c-not-c


TF_Buffer* read_file(const char* file);

void free_buffer(void* data, size_t length) {
        free(data);
}

static void Deallocator(void* data, size_t length, void* arg) {
        free(data);
        // *reinterpret_cast<bool*>(arg) = true;
}

int main() {
  // Use read_file to get graph_def as TF_Buffer*
  TF_Buffer* graph_def = read_file("../trimmed.pb");
  TF_Graph* graph = TF_NewGraph();

  // Import graph_def into graph
  TF_Status* status = TF_NewStatus();
  TF_ImportGraphDefOptions* graph_opts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graph, graph_def, graph_opts, status);
  if (TF_GetCode(status) != TF_OK) {
          fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(status));
          return 1;
  }
  else {
          fprintf(stdout, "Successfully imported graph\n");
  }

  // Create variables to store the size of the input and output variables
  const int num_bytes_in = 2 * sizeof(float);
  const int num_bytes_out = 1 * sizeof(float);

  // Set input dimensions - this should match the dimensionality of the input in
  // the loaded graph, in this case it's three dimensional.
  int64_t in_dims[] = {1,2};
  int64_t out_dims[] = {1,1};

  // ######################
  // Set up graph inputs
  // ######################

  // Create a variable containing your values, in this case the input is a
  // 3-dimensional float
  float values[2] = {-1.04585315e+03,   1.25702492e+02 };

  // Create vectors to store graph input operations and input tensors
  std::vector<TF_Output> inputs;
  std::vector<TF_Tensor*> input_values;

  // Pass the graph and a string name of your input operation
  // (make sure the operation name is correct)
  TF_Operation* input_op = TF_GraphOperationByName(graph, "THEINPUT");
  TF_Output input_opout = {input_op, 0};
  inputs.push_back(input_opout);

  // Create the input tensor using the dimension (in_dims) and size (num_bytes_in)
  // variables created earlier
  TF_Tensor* input = TF_NewTensor(TF_FLOAT, in_dims, 2, values, num_bytes_in, &Deallocator, 0);
  input_values.push_back(input);

  // Optionally, you can check that your input_op and input tensors are correct
  // by using some of the functions provided by the C API.
  std::cout << "Input op info: " << TF_OperationNumOutputs(input_op) << "\n";
  std::cout << "Input data info: " << TF_Dim(input, 0) << "\n";

  // ######################
  // Set up graph outputs (similar to setting up graph inputs)
  // ######################

  // Create vector to store graph output operations
  std::vector<TF_Output> outputs;
  TF_Operation* output_op = TF_GraphOperationByName(graph, "THEMODEL");
  TF_Output output_opout = {output_op, 0};
  outputs.push_back(output_opout);
  #define nullptr NULL
  // Create TF_Tensor* vector
  std::vector<TF_Tensor*> output_values(outputs.size(), nullptr);

  // Similar to creating the input tensor, however here we don't yet have the
  // output values, so we use TF_AllocateTensor()
  TF_Tensor* output_value = TF_AllocateTensor(TF_FLOAT, out_dims, 1, num_bytes_out);
  output_values.push_back(output_value);

  // As with inputs, check the values for the output operation and output tensor
  std::cout << "Output: " << TF_OperationName(output_op) << "\n";
  std::cout << "Output info: " << TF_Dim(output_value, 0) << "\n";

  // ######################
  // Run graph
  // ######################
  fprintf(stdout, "Running session...\n");
  TF_SessionOptions* sess_opts = TF_NewSessionOptions();
  TF_Session* session = TF_NewSession(graph, sess_opts, status);
  assert(TF_GetCode(status) == TF_OK);

  // Call TF_SessionRun
  TF_SessionRun(session, nullptr,
                &inputs[0], &input_values[0], inputs.size(),
                &outputs[0], &output_values[0], outputs.size(),
                nullptr, 0, nullptr, status);

  // Assign the values from the output tensor to a variable and iterate over them
  float* out_vals = static_cast<float*>(TF_TensorData(output_values[0]));
  for (int i = 0; i < 1; ++i)
  {
      std::cout << "Output values info: " << *out_vals++ << "\n";
  }

  fprintf(stdout, "Successfully run session\n");

  // Delete variables
  TF_CloseSession(session, status);
  TF_DeleteSession(session, status);
  TF_DeleteSessionOptions(sess_opts);
  TF_DeleteImportGraphDefOptions(graph_opts);
  TF_DeleteGraph(graph);
  TF_DeleteStatus(status);
  return 0;
}

TF_Buffer* read_file(const char* file) {
  FILE *f = fopen(file, "rb");
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);  //same as rewind(f);

  void* data = malloc(fsize);
  fread(data, fsize, 1, f);
  fclose(f);

  TF_Buffer* buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = free_buffer;
  return buf;
}
