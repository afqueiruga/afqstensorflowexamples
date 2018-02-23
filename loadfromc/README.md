# Loading and running models from C

We want a light-weight library to load into larger applications that only need to execute the tensorflow models.

The static library produced by 
[https://github.com/FloopCZ/tensorflow_cc](https://github.com/FloopCZ/tensorflow_cc)
results in a 100MB executable to just say helloworld! That's too much.
The static library is only for C++ as well, which might be a problem trying to link to a Fortran program.
(Well, to be fair, the problem is Fortran.)
The compilation also didn't work on macOS, yielding unknown instruction errors for the vector operations.
The TensorFlow build scripts don't have a dedicated mac version.
It's not a big deal because I like to run in Docker on mac anyways, but there is another solution.


[https://www.tensorflow.org/install/install_c](https://www.tensorflow.org/install/install_c)



## Loading a graph

The meta graphs don't work in the C API because apparently they have some Python elements. Instead, we have
to serialize the graph into the protobuf description. I had to add a line to the end of the model_loading
notebook to write it out differently,
```python
gd=graph5.as_graph_def()
from tensorflow.python.platform import gfile
with gfile.GFile("trimmed.pb", "w") as f:
      f.write(gd.SerializeToString())
```
	  
