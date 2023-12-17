# CeresLearn
 - Author: chenjingyu
 - 20231216

## 1.项目描述
 - Ceres库学习笔记；

## 2.更新日志
| 日期       | 更新内容                                  |
|----------|---------------------------------------|
| 20231216 | 1.添加HelloWorld测试代码；                   |
| 20231217 | 1.添加一些ceres笔记及Analytic Derivatives代码； |

## 3.项目配置
 - 本项目基于vcpkg包管理、clion编辑器及编译器为MSVC；
```bash
>> vcpkg install ceres[cxsparse,eigensparse,lapack,suitesparse,tools]:x64-windows --recurse
>> vcpkg install opencv:x64-windows
>> vcpkg integrate install
>> CMAKE_TOOLCHAIN_FILE=D:/library/vcpkg/scripts/buildsystems/vcpkg.cmake
```

## 4.参考项目
 - [ceres-solver](https://github.com/ceres-solver/ceres-solver)
