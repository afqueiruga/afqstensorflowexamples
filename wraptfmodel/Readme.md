# A simple wrapper of TF models

We include the definition to allow a nested packed struct instead of 
the opaque pointer paradigm. This removes the dereferencing of the opaque
data pointer when it's already inside of an allocated struct and packs
in our memory tighter.
[There are workarounds for my insane requirements. But let's not go down that road.](https://stackoverflow.com/questions/4440476/static-allocation-of-opaque-data-types)
