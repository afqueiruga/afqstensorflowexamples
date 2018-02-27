#include "stdio.h"
#include "wrappedtfmodel.h"

int main(int argc, char ** argv) {
  WrappedTFModel myModel;
  int status;
  status = WrappedTFModel_Init(&myModel, "../trimmed.pb");
  if(status) return status;
  float input[2] = { 3.0, 1.2 };
  float output[1];
  WrappedTFModel_Eval(&myModel, input,output);
  printf("f(%f,%f) = %f\n",input[0],input[1], output[0]);
  WrappedTFModel_Destroy(&myModel);
  return 0;
}
