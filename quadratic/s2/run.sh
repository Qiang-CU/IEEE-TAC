#!/bin/bash

# 使用 for 循环遍历参数列表
for param in 5 10 15 20 25 30 35 40 45 50
do
    echo "Processing parameter: $param"
    # 在此处执行你需要的操作，例如调用另一个命令或脚本
    mpirun -np $param python main_s2.py $param
done