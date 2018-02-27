#include "tensorflow/c/c_api.h"

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
#define INP_MAX 1
#define OUT_MAX 1
typedef  double real_t;
typedef struct pop_model_str {
  TF_Graph* graph;
  TF_Input inputs[INP_MAX];
  TF_Tensor * in_tens;
  TF_Output outputs[OUT_MAX];
  TF_Tensor * out_tens;
  
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
  TF_GraphImportGraphDef(self->graph, graph_def, graph_opts, self->status);
  if (TF_GetCode(self->status) != TF_OK) {
	fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(self->status));
	return -1;
  }
  else {
	fprintf(stdout, "Successfully imported graph\n");
  }
  TF_DeleteImportGraphDefOptions(graph_opts);
  TF_DeleteBuffer(graph_def);

  // Extract the input and output operations
  TF_Operation* input_op = TF_GraphOperationByName(self->graph, "THEINPUT");
  self->inputs[0].oper = input_op;
  self->inputs[0].index = 0;
  TF_Operation* output_op = TF_GraphOperationByName(self->graph, "THEMODEL");
  self->outputs[0].oper = output_op;
  self->outputs[0].index = 0;

  
  TF_AttrMetadata inmeta = TF_OperationGetAttrMetadata(input_op,"shape", self->status);
  int nidims = inmeta.total_size;
  printf("num dims is %d\n",nidims);
  int64_t idims[nidims];
  TF_OperationGetAttrShape(input_op,"shape", idims,nidims,self->status);
  printf("the dims are %ld %ld\n",idims[0],idims[1]);
  if(idims[0]==-1) idims[0]=1;
  size_t num_bytes_in=1, num_bytes_out=1;
  for(int i=0;i<nidims;i++) num_bytes_in*=idims[i];
  num_bytes_in*= sizeof(float);
  
  int nodims = TF_GraphGetTensorNumDims(self->graph,self->outputs[0],self->status);
  printf("%d\n",nodims);
  int64_t odims[nodims];
  TF_GraphGetTensorShape(self->graph,self->outputs[0],odims,nodims,self->status);
  printf("%ld %ld\n",odims[0],odims[1]);
  if(odims[0]==-1) odims[0]=1;
  for(int i=0;i<nodims;i++) num_bytes_out*=odims[i];
  num_bytes_out *= sizeof(float);
  // We allocate the tensors ahead of time and then copy into them
  self->in_tens = TF_AllocateTensor(TF_FLOAT, idims,  nidims, num_bytes_in);
  self->out_tens= TF_AllocateTensor(TF_DOUBLE, odims, nodims, num_bytes_out);
  
  // Set up a session which will be used to call this graph
  self->opts = TF_NewSessionOptions();
  self->sess = TF_NewSession(self->graph, self->opts, self->status);
  if(TF_GetCode(self->status) != TF_OK) {
	fprintf(stderr, "ERROR: Unable to create session %s",TF_Message(self->status));
	return -1;
  }

}

void PopModel_Destroy(pop_model_t * self) {
  TF_CloseSession(self->sess, self->status);
  TF_DeleteSession(self->sess, self->status);
  TF_DeleteSessionOptions(self->opts);
  TF_DeleteGraph(self->graph);
  TF_DeleteStatus(self->status);
}

void PopModel_Eval(pop_model_t * self, real_t * input, real_t * output) {
  // This routine needs to be made AS FAST AS POSSIBLE
  
}


int main(int argc, char ** argv) {
  pop_model_t myModel;
  PopModel_Init(&myModel, "../trimmed.pb");

  PopModel_Destroy(&myModel);
  return 0;
}
