gcc -I/home/afq/opt/tensorflowcapi/include -L/home/afq/opt/tensorflowcapi/lib load_graph.c -ltensorflow -o load_graph.out
g++ -g -I/home/afq/opt/tensorflowcapi/include -L/home/afq/opt/tensorflowcapi/lib runit.cpp -ltensorflow -o runit.out
