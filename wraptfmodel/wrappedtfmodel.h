#ifndef WRAPTFMODEL_H_
#define WRAPTFMODEL_H_

#define REAL float
typedef struct WrappedTFModel WrappedTFModel;

int  WrappedTFModel_Init(WrappedTFModel * self, char * fname);
void WrappedTFModel_Destroy(WrappedTFModel * self);
void WrappedTFModel_Eval(WrappedTFModel * self, REAL * input, REAL * output);


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
