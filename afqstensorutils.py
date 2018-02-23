import numpy as np
import tensorflow as tf

def Divy(data, xslice, yslice, rtest, rvalid=0):
    """
    Divy data up into groups for training
    """
    # Prep the training data
    n = data.shape[0]
    idxs = np.array(range(n)).astype(np.int32)
    np.random.shuffle(idxs)
    
    nTestIdxs = int(n * rtest)
    nValidIdxs = int(n * rvalid)
    validIdxs = idxs[0:nValidIdxs]
    testIdxs = idxs[nValidIdxs:nValidIdxs + nTestIdxs]
    nTrainIdxs = n - nValidIdxs - nTestIdxs
    trainIdxs = idxs[nValidIdxs + nTestIdxs:n]
    print('Training data points: %d' % nTrainIdxs)
    print('Testing data points: %d' % nTestIdxs)
    print('Validation data points: %d' % nValidIdxs)
    train_x = data[trainIdxs,xslice]
    test_x  = data[testIdxs, xslice]
    valid_x = data[validIdxs,xslice]
    train_y = data[trainIdxs,yslice]
    test_y  = data[testIdxs, yslice]
    valid_y = data[validIdxs,yslice]
    if len(train_x.shape)<2:
        train_x = train_x.reshape(train_x.size,1)
        test_x = test_x.reshape(test_x.size,1)
        valid_x = valid_x.reshape(valid_x.size,1)
    if len(train_y.shape)<2:
        train_y = train_y.reshape(train_y.size,1)
        test_y = test_y.reshape(test_y.size,1)
        valid_y = valid_y.reshape(valid_y.size,1)
        
    return train_x,train_y, test_x,test_y, valid_x,valid_y

class Scaler():
    """
    Helper class to manage scaling a neural network to the nominal range.
    TODO: Tensorflowify this to apply it to a model automagically.
    """
    def __init__(self, data):
        self.scale = []
        ndata = np.empty(data.shape)
        for i in xrange(data.shape[1]):
            x = data[:,i]
            self.scale.append( [np.mean(x), np.amin(x), np.amax(x)] )
            ndata[:,i] = (x - self.scale[-1][0]) /\
              (self.scale[-1][2] - self.scale[-1][1])
        self.scaled_data = ndata
        
    def apply(self, x, i):
        y = np.empty(x.shape)
        for j in xrange(x.shape[-1]):
            y[:,j] = (x[:,j] - self.scale[i+j][0] ) /\
              ( self.scale[i+j][2]-self.scale[i+j][1] )
        return y
    
    def invert(self, x, i):
        y = np.empty(x.shape)
        for j in xrange(x.shape[-1]):
            y[:,j] = x[:,j] * ( self.scale[i+j][2]-self.scale[i+j][1] ) \
              + self.scale[i+j][0]
        return y

def CatVariable(shapes, stddev=0.0):
    l = 0
    for shp in shapes:
        il = 1
        for s in shp: il *= s
        l += il
    # V = tf.Variable(tf.zeros(shape=(l,)))
    V = tf.Variable(tf.truncated_normal(shape=(l,), stddev=stddev))
    cuts = []
    l = 0
    for shp in shapes:
        il = 1
        for s in shp: il *= s
        cuts.append(tf.reshape(V[l:(l+il)],shp))
        l += il
    return V, cuts

def NewtonsMethod(P, x, alpha=1.0):
    """
    Gives you an operator that performs standard Newton's method
    """
    if len(x.shape)!=1:
        Exception('')
    N = x.shape[0]
    Grad = tf.gradients(P,x)[0]
    # TensorFlow f's up the shapes a lot, so I'm constantly reshaping
    Hess = tf.reshape(tf.hessians(P,x)[0],shape=[N,N])
    return [
      x.assign_add(-tf.squeeze( # Reshaping to have 1 dimension
                   # Never invert!!!!
                   tf.matrix_solve(Hess,
                            tf.expand_dims(Grad,1))# Reshaping to have 2 dimensions
                ))
    ]
    
def outer(a,b, triangle=False):
    """
    Symbolic outer product:
    stack( a(x)b )
    triangle option toggles whether or not to include symmetric elements (i.e. 
    only return the lower triangle because it's identical to the upper triangle)
    You probably want triangle=True when a==b.
    """
    p = []
    for i in xrange(a.shape[-1]):
        for j in xrange(i if triangle else 0,b.shape[-1]):
            p.append( a[:,i]*b[:,j] )
    return tf.stack(p, axis=-1)

def polyexpand(a,o):
    """
    Build and stack a polynomial basis set of a up to exponent o. Includes
    all cross terms. E.g.,
    polyexpand([x y], 2) = [ x y x^2 xy y^2 ]
    """
    if o<=0: raise Exception("I don't know what it means when o<=0")
    if o==1: return a
    p = [a,outer(a,a,True)]
    for i in xrange(2,o+1):
        p.append(outer(p[-1],a))
    return tf.concat(p, axis=1)

def travel_op(op,indents=0):
    "Print a tree from a tensorflow op"
    # TODO: Detect op or tensor
    for _ in xrange(indents): print " ",
    print op.name, " : ", op.type
    for i in op.inputs:
        travel_op(i.op,indents+2)
        
def write_trimmed_meta_graph(graph,sess,
                             outputs,# input_node,
                             ofile):
    """
    Take a graph description, trim it down, and write it
    """
    output_graph_def \
        = tf.graph_util.convert_variables_to_constants(
            sess,graph.as_graph_def(),outputs)
    sub_output = tf.graph_util.extract_sub_graph(
        output_graph_def, outputs)
    newgraph=tf.Graph()
    with newgraph.as_default():
        tf.import_graph_def(sub_output,
                            name='',
                            return_elements=outputs)
        meta_graph_def = tf.train.export_meta_graph(filename=ofile,
                                                    graph_def=newgraph.as_graph_def())

        
def write_trimmed_pb_graph(graph,sess,
                           outputs,# input_node,
                           ofile):
    """
    Take a graph description, trim it down, and write it
    """
    output_graph_def \
        = tf.graph_util.convert_variables_to_constants(
            sess,graph.as_graph_def(),outputs)
    sub_output = tf.graph_util.extract_sub_graph(
        output_graph_def, outputs)

    from tensorflow.python.platform import gfile
    with gfile.GFile(ofile, "wb") as f:
        f.write(sub_output.SerializeToString())
