#include "tensororflow/c/c_api.h"

#include <stdio.h>
#include <stdlib.h>

/* This is a prototype library for model loading */

/* These are the helper files */
TF_Buffer* read_file(const char* file);
void free_buffer(void* data, size_t length);
static void Deallocator(void* data, size_t length, void* arg);

TF_Buffer* read_file(const char* file) {
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
void free_buffer(void* data, size_t length) {
        free(data);
}
static void Deallocator(void* data, size_t length, void* arg) {
        free(data);
}

/* My model library */

typedef struct pop_model_str {
  TF_Graph* graph;
  TF_Output* inputs;
  TF_Output* outputs;

  TF_Status * status;
  TF_SessionOptions * opts;
  TF_Session* sess;
} pop_model_t;

int PopModel_Init(pop_model_t * self, char * fname) {
  // Read the protobuf file
  TF_Buffer* graph_def = read_file(fname);
  if(!graph_def) return -1;

  // Construct the graph
  self->status = TF_NewStatus();
  self->graph = TF_NewGraph();
  TF_ImportGraphDefOptions* graph_opts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graph, graph_def, graph_opts, status);
  if (TF_GetCode(status) != TF_OK) {
          fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(status));
          return 1;
  }
  else {
          fprintf(stdout, "Successfully imported graph\n");
  }
  TF_DeleteImportGraphDefOptions(graph_opts);

  // Set up a session which will be used to call this graph
  opts = TF_NewSessionOptions();
  self->sess = TF_NewSession(self->graph, self->opts, self->status);
}

void PopModel_Destroy(pop_model_t * self) {
  TF_CloseSession(self->sess, self->status);
  TF_DeleteSession(self->sess, self->status);
  TF_DeleteSessionOptions(self->opts);
  
}

void PopModel_Eval(pop_model_t * self, real_t * input, real_t * output) {
  // This routine needs to be made AS FAST AS POSSIBLE
}
