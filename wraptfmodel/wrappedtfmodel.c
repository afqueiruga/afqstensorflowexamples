#include "wrappedtfmodel.h"


#include <stdio.h>
#include <stdlib.h>

static void free_buffer(void* data, size_t length) {
        free(data);
}
static TF_Buffer* read_file(const char* file) {
  FILE *f = fopen(file, "rb");
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);
  void* data = malloc(fsize);
  fread(data, fsize, 1, f);
  fclose(f);
  TF_Buffer* buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = free_buffer;
  return buf;
}


int  WrappedTFModel_Init(WrappedTFModel * self, char * fname) {

}
void WrappedTFModel_Destroy(WrappedTFModel * self) {

}
void WrappedTFModel_Eval(WrappedTFModel * self, REAL * input, REAL * output) {

}
