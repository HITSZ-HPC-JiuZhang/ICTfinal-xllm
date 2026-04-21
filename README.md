推理模型性能优化
目标：在确保功能正确的情况下，获得模型最高优化性能
模型：Qwen3.5-9B
推理框架：xLLM ict_final分支

要求：在保证精度的前提下（详情见xLLM 精度测试），使用各种优化方法，优化Qwen3.5-9B的推理性能，尽可能提高 Output Tokens per Second (输出Tokens/秒， TPS)：
测试1：单并发，输入输出64k+1k，比拼TPS
测试2：单并发，输入输出128k+1k，比拼TPS
排名结果比拼测试1、测试2的TPS之和

最终提交的报告至少需要包括以下部分：
1）优化方法介绍
2）精度测试结果
3）性能测试结果
其他要求：需要将代码和编译完成的二进制xllm文件一并提交


模型：已安装 /root/xllm路径下
权重文件：/home/aicc/IctModel/
测试数据集文件：/home/aicc/IctModel/
首次执行编译前，请先执行如下命令：
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
并将编译命令修改为：VCPKG_FORCE_SYSTEM_BINARIES=1 VCPKG_ROOT=/vcpkg-src/ python setup.py build
