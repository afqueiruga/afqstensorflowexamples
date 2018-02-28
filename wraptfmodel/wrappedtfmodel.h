#ifndef WRAPTFMODEL_H_
#define WRAPTFMODEL_H_

#ifndef WRAP_C_TYPE
#define WRAP_C_TYPE float
#endif
#ifndef WRAP_TF_TYPE
#define WRAP_TF_TYPE TF_FLOAT
#endif
typedef struct WrappedTFModel WrappedTFModel;

int  WrappedTFModel_Init(WrappedTFModel * self, char * fname,
						 char * input_op_name, char * output_op_name);
void WrappedTFModel_Destroy(WrappedTFModel * self);
void WrappedTFModel_Eval(WrappedTFModel * self, WRAP_C_TYPE * input, WRAP_C_TYPE * output);


/* Opaque pointers are meh. Complete the struct definition to allow the object
   to live on the stack or in a struct.
*/
#include "tensorflow/c/c_api.h"
struct WrappedTFModel {
  TF_Graph* graph;

  TF_Output input;
  TF_Tensor * in_tens;
  size_t in_length;
  
  TF_Output output;
  TF_Tensor * out_tens;
  size_t out_length;
  
  TF_Status * status;
  TF_SessionOptions * opts;
  TF_Session* sess;
};



#endif
