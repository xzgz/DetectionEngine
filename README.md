# Implement the inference process of ObjectFlow in C++.

# Install:
Before compile, we need to install the tools for the building of protobuf:
```shell
sudo yum install autoconf automake libtool curl make g++ unzip
```

For the first time, we need to build the dependencies of CObjectFlow, use:
```shell
sh compile_with_external.sh
```
After that, if we need to build CObjectFlow again, we only need to build the source of CObjectFlow alone:
```shell
sh compile.sh
```

