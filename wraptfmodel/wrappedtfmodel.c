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


int  WrappedTFModel_Init(WrappedTFModel * self, char * fname,
						 char * input_op_name, char * output_op_name) {
  // Read the protobuf file
  printf("WrappedTFModel: reading the graph in %s\n",fname);
  TF_Buffer* graph_def = read_file(fname);
  if(!graph_def) {
	printf("ERROR: Could not read from file.\n");
	return -1;
  }

  // Construct the graph
  self->status = TF_NewStatus();
  self->graph = TF_NewGraph();
  TF_ImportGraphDefOptions* graph_opts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(self->graph, graph_def, graph_opts, self->status);
  if (TF_GetCode(self->status) != TF_OK) {
	fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(self->status));
	return -1;
  }
  TF_DeleteImportGraphDefOptions(graph_opts);
  TF_DeleteBuffer(graph_def);

  // Extract and set up the input
  TF_Operation* input_op = TF_GraphOperationByName(self->graph, input_op_name);
  self->input.oper = input_op;
  self->input.index = 0;
  TF_AttrMetadata inmeta = TF_OperationGetAttrMetadata(input_op,"shape", self->status);
  int nidims = inmeta.total_size;
  int64_t idims[nidims];
  TF_OperationGetAttrShape(input_op,"shape", idims,nidims,self->status);
  printf("The input dims are ("); for(int i=0;i<nidims-1;i++) printf("%ld, ",idims[i]);
  printf("%ld)\n",idims[nidims-1]);
  if(idims[0]==-1) idims[0]=1;
  size_t num_bytes_in=1, num_bytes_out=1;
  for(int i=0;i<nidims;i++) num_bytes_in*=idims[i];
  self->in_length = num_bytes_in;
  num_bytes_in*= sizeof(WRAP_C_TYPE);
  self->in_tens = TF_AllocateTensor(WRAP_TF_TYPE, idims,  nidims, num_bytes_in);

  // Extract and set up the output
  TF_Operation* output_op = TF_GraphOperationByName(self->graph, output_op_name);
  self->output.oper = output_op;
  self->output.index = 0;
  int nodims = TF_GraphGetTensorNumDims(self->graph,self->output,self->status);
  int64_t odims[nodims];
  TF_GraphGetTensorShape(self->graph,self->output,odims,nodims,self->status);
  printf("The output dims are ("); for(int i=0;i<nodims-1;i++) printf("%ld, ",odims[i]);
  printf("%ld)\n",odims[nodims-1]);
  if(odims[0]==-1) odims[0]=1;
  for(int i=0;i<nodims;i++) num_bytes_out*=odims[i];
  self->out_length = num_bytes_out;
  num_bytes_out *= sizeof(WRAP_C_TYPE);
  self->out_tens= TF_AllocateTensor(WRAP_TF_TYPE, odims, nodims, num_bytes_out);

  // Set up a session which will be used to call this graph
  self->opts = TF_NewSessionOptions();
  self->sess = TF_NewSession(self->graph, self->opts, self->status);
  if(TF_GetCode(self->status) != TF_OK) {
	fprintf(stderr, "ERROR: Unable to create session %s",TF_Message(self->status));
	return -1;
  }
  return 0;
}

void WrappedTFModel_Destroy(WrappedTFModel * self) {
  TF_CloseSession(self->sess, self->status);
  TF_DeleteSession(self->sess, self->status);
  TF_DeleteSessionOptions(self->opts);
  TF_DeleteGraph(self->graph);
  TF_DeleteStatus(self->status);
}
void WrappedTFModel_Eval(WrappedTFModel * self, WRAP_C_TYPE * input, WRAP_C_TYPE * output) {
  WRAP_C_TYPE * ti = TF_TensorData(self->in_tens);
  for(int i=0;i<self->in_length;i++) ti[i] = input[i];
  TF_SessionRun(self->sess, NULL,
                &self->input, &self->in_tens, 1,
                &self->output, &self->out_tens, 1,
                NULL, 0, NULL, self->status);
  WRAP_C_TYPE * to = TF_TensorData(self->out_tens);
  for(int i=0;i<self->out_length;i++) output[i] = to[i];
}
