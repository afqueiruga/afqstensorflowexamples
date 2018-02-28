# A simple wrapper of TF models

## Setting up the model

## Design of the wrapper

We include the definition to allow a nested packed struct instead of 
the opaque pointer paradigm. This removes the dereferencing of the opaque
data pointer when it's already inside of an allocated struct and packs
in our memory tighter.
[There are workarounds for my insane requirements. But let's not go down that road.](https://stackoverflow.com/questions/4440476/static-allocation-of-opaque-data-types)

## Using and compiling the wrapper

```bash
gcc -I/home/afq/opt/tensorflowcapi/include
-L/home/afq/opt/tensorflowcapi/lib wrappedtfmodel.c main.c -ltensorflow -o test.out
```
