# xLLM 运行与排障笔记

python test_xllm.py \
    --backend xllm \
    --dataset-name random \
    --random-range-ratio 1 \
    --num-prompt 1 \
    --max-concurrency 1 \
    --random-input  65536 \
    --random-output 1024 \
    --host 127.0.0.1 \
    --port 18000 \
    --dataset-path /home/kuma/project/ict-final/xllm/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
    --model /mnt/nvme0/models/Qwen3.5/Qwen3.5-4B

 curl -s "http://127.0.0.1:18000/v1/chat/completions"     -H "Content-Type: application/json"     -H "Authorization: Bearer <API Key>"     -d '{
          "model": "Qwen3.5-4B",
          "messages": [
            {"role": "system", "content": "You are a user assistant."},
            {"role": "user", "content": "介绍下北京"}
          ],
          "top_p": 0.95,
          "max_tokens": 1024,
          "temperature": 0.6,
          "top_k": -1,
          "stream": false
        }'


## 1. AGENTS.md 临时保护与恢复

```bash
mv AGENTS.md /tmp/AGENTS.mine.md
git pull --rebase
cp /tmp/AGENTS.mine.md AGENTS.md
git update-index --skip-worktree AGENTS.md
git ls-files -v -- AGENTS.md
/
```

### 取消 skip-worktree

```text
git update-index --no-skip-worktree AGENTS.md
```

## 2. NPU 设备检查

```bash
python -c "import torch_npu
for i in range(8):torch_npu.npu.set_device(i)"
```

## 3. 终止服务

### 3.1 检查线程

```bash
pgrep - af xllm
```

### 3.2 停止

```bash
sudo pkill - 9 xllm
```

### 3.3 再次检查停止

```bash
pgrep - af '/build/xllm/core/server/xllm' ||
    echo stopped
```

### 3.4 设置 skip-worktree

```text
git update -
index-- skip -
worktree AGENTS.md
```

## 4. tmux 常用操作

功能,
命令 / 快捷键 新建会话, tmux new - s<name> 分离当前会话(回到原生终端),
Prefix + d 查看所有会话, tmux ls 接入指定会话,
tmux attach - t<name> 重命名当前会话, Prefix + $ 杀死指定会话,
tmux kill - session -
t<name>

## 5. 启动命令

### 5.1 启动

```text
E bash run_qwen.sh
```

### 5.2 基准测试

```text
python3 tools
/ benchmark_inference.py-- api chat-- model Qwen3
- Next - 80B - A3B -
Instruct-- duration 60 --concurrency 2 --timeout 60 --qps 1 --prompt
"hello qwen"
```

## 6. 在线服务（chat 模式）

```bash
curl -s "http://127.0.0.1:18000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <API Key>" \
    -d '{
          "model": "Qwen3.5-4B",
          "messages": [
            {"role": "system", "content": "You are a user assistant."},
            {"role": "user", "content": "介绍下北京"}
          ],
          "top_p": 0.95,
          "temperature": 0.6,
          "top_k": -1,
          "stream": false
        }'
```

```bash
curl http :  //localhost:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
                      "model" : "Qwen3-30B-A3B-Thinking-2507",
    "max_tokens" : 1000, "temperature" : 0.5, "stream" : false, "messages" : [
      {"role" : "system", "content" : "You are a helpful assistant."},
      {"role" : "user", "content" : "你好 xllm"}
    ]
}
'
```

## 7. Docker 内运行流程说明

我的运行环境在docker内，你需要先执行 docker exec -it --user kuma xllm-npu ，以kuma 用户身份进入容器 cd project/xllm ，清理进程 sudo pkill -9 xllm , 重新编译 sudo python setup.py build , sudo bash run_llada2_1_mini.sh 启动xllm，然后发送请求（特别说明 sudo无需密码，底层依赖可能需要sudo权限）

### 7.1 请求示例（LLaDA2.1-flash）

```bash
curl http://localhost:18000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "LLaDA2.1-flash",
      "temperature": 0.0,
      "max_tokens": 128,
      "stream": false,
      "messages": [
        {
          "role": "user",
          "content": "你好,你是谁"
        }
      ]
    }'
```

### 7.2 请求1

```bash
curl -X POST "http://127.0.0.1:18000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LLaDA2.1-mini",
    "stream": false,
    "temperature": 0,
    "max_tokens": 1024,
    "messages": [
      {"role": "system", "content": "detailed thinking off"},
      {"role": "user", "content": "Write the number from 1 to 128"}
    ]
  }'
```

### 7.3 请求2

```bash
curl http://localhost:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LLaDA2.1-mini",
    "max_tokens": 64,
    "stream": false,
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "肯德基有什么好吃的"
      }
    ]
  }'
```

## 8. 环境变量与量化命令

```text
export ASCEND_RT_VISIBLE_DEVICES = 0,
       1, 2, 3, 4, 5, 6,
       7 export PYTORCH_NPU_ALLOC_CONF =
           expandable_segments : False

                                     cd /
                                 home /
                                 kuma /
                                 project /
                                 msmodelslim /
                                 example /
                                 Qwen3 -
           MOE

                   python3 quant_qwen_moe_w8a8.py-- model_path /
               home / kuma / model / Qwen3 / Qwen3 -
           30B - A3B - Thinking -
           2507 --save_path / home / kuma / model / Qwen3 / Qwen3 - 30B -
           A3B - Thinking - 2507 - w8a8-- anti_dataset../ common / qwen3 -
           moe_anti_prompt_50.json-- calib_dataset../ common / qwen3 -
           moe_calib_prompt_50.json-- trust_remote_code
               True-- batch_size 1 --device_type npu : 0,
       1, 2, 3, 4, 5, 6,
       7 --rot
```

## 9. NPU 后端兼容说明

在不带 USE_NPU_TORCH 的常规 NPU 构建中，可能出现 model type recognized 失败

```cpp
#if defined(USE_NPU)
    if (model_type == "qwen3_next" && FLAGS_npu_kernel_backend != "TORCH") {
  LOG(WARNING) << "qwen3_next requires npu_kernel_backend=TORCH; "
                  "overriding from "
               << FLAGS_npu_kernel_backend << " to TORCH.";
  FLAGS_npu_kernel_backend = "TORCH";
}
#endif
```

---

## 2026-03-27 变更记录（xllm.cpp）

### 问题
- `xllm/xllm.cpp` 中模型名提取在两处重复（约 156、364 行附近）。
- 条件判断使用归一化路径 `model_path`，但取值重新从原始 `FLAGS_model` 解析，存在边界路径行为不一致风险。

### 修改
- 新增统一提取函数：`extract_model_name_from_path(const std::filesystem::path&)`。
- 在 `run()` 中仅基于 `model_path = std::filesystem::path(FLAGS_model).lexically_normal()` 提取一次 `default_model_name`。
- `FLAGS_model_id`（仅在未显式指定时）与 `model_version` 统一复用该结果。

### 兼容性说明（保持不变）
- 未修改 gflags 解析流程：`google::ParseCommandLineFlags(&argc, &argv, true)`。
- 未修改 `--model` flag 定义、名称、必填校验逻辑。
- 仍兼容原有传参方式：
  - `--model "/path/to/model"`
  - `--model="/path/to/model"`
  - 含空格路径（通过 shell 引号传入）
- 显式 `--model_id` 仍优先，不受影响。

### 影响范围
- 仅改动：`xllm/xllm.cpp`

---

## 2026-03-27 Change Summary (English)

### Problem
- In `xllm/xllm.cpp`, model name extraction logic was duplicated in two places (around lines 156 and 364).
- The condition used normalized `model_path`, while value extraction re-parsed raw `FLAGS_model`, which could cause inconsistent behavior on edge-case paths.

### What was changed
- Added a shared helper: `extract_model_name_from_path(const std::filesystem::path&)`.
- In `run()`, compute `default_model_name` once from normalized `model_path`.
- Reused the same value for:
  - default `FLAGS_model_id` (only when `--model_id` is not explicitly set)
  - `model_version`

### Compatibility
- No change to gflags parsing flow: `google::ParseCommandLineFlags(&argc, &argv, true)`.
- No change to `--model` flag name, type, or required-check behavior.
- Existing CLI usage remains compatible:
  - `--model "/path/to/model"`
  - `--model="/path/to/model"`
  - quoted paths with spaces
- Explicit `--model_id` still takes precedence.

### Scope
- Modified file: `xllm/xllm.cpp`

---

## 2026-03-27 冲突解决与函数下沉记录（新增）

### 背景
- `xllm/xllm.cpp` 存在未解决的 merge conflict，冲突块中包含本地 `extract_model_name_from_path` 与本地 `get_model_backend` 实现。
- 其中 `get_model_backend` 与 `xllm/core/util/utils.h` 已有实现重复。

### 处理结果
- 已清理 `xllm/xllm.cpp` 冲突标记并完成冲突解决。
- `extract_model_name_from_path` 已重构下沉到 `xllm/core/util/utils.h`（`xllm::util` 命名空间）。
- `run()` 中改为调用 `xllm::util::extract_model_name_from_path(model_path)`。
- 保持与上次提交一致：
  - 基于 `lexically_normal()` 后的 `model_path` 提取一次 `default_model_name`
  - 同时用于默认 `FLAGS_model_id` 与 `model_version`
- 删除了 `xllm/xllm.cpp` 中重复的本地 backend 解析逻辑，统一走 `xllm::util::get_model_backend(model_path)`。

### 本次影响文件
- `xllm/xllm.cpp`
- `xllm/core/util/utils.h`

### 代码审查（skill）
- 已调用项目 `code-review` skill 执行审查流程。

## 10. Docker 启动（简版）

```bash
docker run -it \
--ipc=host \
-u 0 \
--name xllm \
--privileged \
--network=host \
--device=/dev/davinci0 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/sbin/:/usr/local/sbin/ \
-v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
-v /var/log/npu/slog/:/var/log/npu/slog \
-v /var/log/npu/profiling/:/var/log/npu/profiling \
-v /var/log/npu/dump/:/var/log/npu/dump \
-v $HOME:$HOME \
-w $HOME \
quay.nju.edu.cn/jd_xllm/xllm-ai/xllm-dev-a2-arm-20260306 \
/bin/bash
```

## 11. 数据盘权限说明

是的，**完全可以**，而且这是**最推荐、最直接**的做法。

在 Linux 系统中，你可以通过修改文件或目录的 **所有者（Owner）** 和 **权限位（Permission Bits）**，让普通用户（如 `kuma`）拥有对数据盘的读写权限。

---

### ✅ 推荐做法：将数据盘所有权赋予 `kuma` 用户

假设你的数据盘挂载在宿主机的 `/mnt/nvme0`，并且 `kuma` 用户的 UID 是 `1001`（请先用 `id kuma` 确认），执行以下命令：

```bash
# 1. 确认 kuma 的 UID 和 GID
id kuma
# 输出示例：uid=1001(kuma) gid=1001(kuma)

# 2. 修改整个数据盘目录的所有者为 kuma
sudo chown -R 1001:1001 /mnt/nvme0

# 3. （可选但推荐）确保有完整的读写权限
sudo chmod -R u+rwX /mnt/nvme0
```

> 🔍 **说明**：
> - `-R`：递归处理所有子目录和文件。
> - `u+rwX`：给“所有者”（user）添加读（r）、写（w）权限；`X` 表示仅对目录或已有执行权限的文件添加执行权限（安全）。
> - 这样操作后，`kuma` 用户（UID=1001）就能像操作自己家目录一样读写 `/mnt/nvme0` 下的所有内容。

---

### ⚠️ 注意事项

#### 1. **不要直接 `chmod 777`**
虽然 `sudo chmod -R 777 /mnt/nvme0` 能让所有用户读写，但这会带来严重的**安全风险**（任何用户甚至服务都能篡改或删除你的模型文件）。**强烈不建议**。

#### 2. **如果是多用户共享，考虑使用组（Group）**
如果你有多个用户（如 `kuma`, `alice`）都需要访问这块盘：
```bash
# 创建一个共享组
sudo groupadd model_users

# 将用户加入组
sudo usermod -aG model_users kuma
sudo usermod -aG model_users alice

# 设置目录归属到该组
sudo chgrp -R model_users /mnt/nvme0
sudo chmod -R g+rwX /mnt/nvme0

# 设置 setgid 位，确保新创建的文件自动继承组
sudo chmod g+s /mnt/nvme0
```

#### 3. **云服务器数据盘的初始状态**
- 新购买的云硬盘（如阿里云 ESSD、华为云 EVS）在首次挂载后通常是空的，且默认属主是 `root`。
- 只要你没有在上面运行过其他服务（如数据库、Web 服务），直接 `chown` 给 `kuma` 是完全安全的。

#### 4. **Docker 容器内权限一致性**
当你在宿主机上把 `/mnt/nvme0` 的所有者设为 UID=1001 后：
- 在容器内以 `-u 1001:1001` 启动时，该用户就能无缝读写。
- 即使容器内没有叫 `kuma` 的用户（只有 UID），Linux 内核也只认 UID/GID，所以权限依然生效。

---

### 🧪 验证是否成功

```bash
# 在宿主机切换到 kuma 用户
sudo -u kuma touch /mnt/nvme0/test_file && echo "✅ 写入成功" || echo "❌ 权限不足"

# 清理测试文件
sudo -u kuma rm /mnt/nvme0/test_file
```

如果输出 “✅ 写入成功”，说明权限已正确配置。

---

### 总结

| 操作 | 命令 | 是否推荐 |
|------|------|--------|
| **赋权给指定用户** | `sudo chown -R 1001:1001 /mnt/nvme0` | ✅ 强烈推荐 |
| **开放所有权限** | `sudo chmod -R 777 /mnt/nvme0` | ❌ 不安全，禁止 |
| **使用共享组** | `chgrp + chmod g+rwX` | ✅ 多用户场景推荐 |

**结论**：放心地用 `chown` 把数据盘交给 `kuma` 用户，这是标准且安全的做法。

根据你提供的信息，你的宿主机上有多个 NVMe 数据盘（`/mnt/nvme0`, `/mnt/nvme1`, `/mnt/nvme2`），而你当前的 Docker 容器是以 **root 用户**（`-u 0`）启动的，并且只挂载了 `$HOME`（即 `/home/kuma`）。

你的目标是：
1. **以 `kuma` 用户身份运行容器**（避免权限问题）。
2. **将数据盘（如 `/mnt/nvme0`）挂载到容器内**，以便在容器中访问模型等大文件。

---

### ✅ 解决方案：修改 `docker run` 命令

你需要做两件事：
1. **移除 `-u 0`，改用 `kuma` 的 UID/GID**（通常是 `1001` 或 `1002`，需确认）。
2. **添加 `-v /mnt/nvme0:/mnt/nvme0`**（或其他你想挂载的数据盘）。

#### 步骤 1：确认 `kuma` 用户的 UID 和 GID
在宿主机上运行：
```bash
id kuma
```
假设输出是：
```
uid=1001(kuma) gid=1001(kuma) groups=1001(kuma),...
```
那么 UID=1001, GID=1001。

> ⚠️ 注意：从你的 `df -h` 输出看，`/run/user/1001` 和 `/run/user/1002` 都存在，说明可能有多个用户。请务必用 `id kuma` 确认。

## Docker 启动

```bash
# 启动容器

docker run -it \
--ipc=host \
--name xllm \
--privileged \
--network=host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/sbin/:/usr/local/sbin/ \
-v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
-v /var/log/npu/slog/:/var/log/npu/slog \
-v /var/log/npu/profiling/:/var/log/npu/profiling \
-v /var/log/npu/dump/:/var/log/npu/dump \
-v $HOME:$HOME \
-v /mnt/nvme0:/mnt/nvme0 \
-v /mnt/nvme1:/mnt/nvme1 \
-v /mnt/nvme2:/mnt/nvme2 \
-u 0 \
-w $HOME \
quay.nju.edu.cn/jd_xllm/xllm-ai:xllm-dev-a2-arm-20260306 \
/bin/bash
```

### apptainer
```bash
apptainer exec \
--fakeroot \
--pwd $HOME \
--bind /dev/davinci0:/dev/davinci0 \
--bind /dev/davinci1:/dev/davinci1 \
--bind /dev/davinci2:/dev/davinci2 \
--bind /dev/davinci3:/dev/davinci3 \
--bind /dev/davinci4:/dev/davinci4 \
--bind /dev/davinci5:/dev/davinci5 \
--bind /dev/davinci6:/dev/davinci6 \
--bind /dev/davinci7:/dev/davinci7 \
--bind /dev/davinci_manager:/dev/davinci_manager \
--bind /dev/devmm_svm:/dev/devmm_svm \
--bind /dev/hisi_hdc:/dev/hisi_hdc \
--bind /usr/local/Ascend/driver:/usr/local/Ascend/driver \
--bind /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
--bind /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
--bind /usr/local/sbin/:/usr/local/sbin/ \
--bind /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
--bind /var/log/npu/slog/:/var/log/npu/slog \
--bind /var/log/npu/profiling/:/var/log/npu/profiling \
--bind /var/log/npu/dump/:/var/log/npu/dump \
--bind $HOME:$HOME \
--bind /mnt/nvme0:/mnt/nvme0 \
--bind /mnt/nvme1:/mnt/nvme1 \
--bind /mnt/nvme2:/mnt/nvme2 \
/mnt/nvme0/containers/xllm-dev-a2-arm-20260306.sif \
/bin/bash
```
```bash

```

### 权限问题

#### 容器内创建相同用户

- root shell 创建用户
```bash
# openEuler 系统（基于 RHEL/CentOS）
# 以用户 kuma 为例

groupadd -g 1001 kuma
useradd -u 1001 -g 1001 -d /home/kuma -m -s /bin/bash kuma
```

- 安装 sudo 并赋权免密
```bash
# 安装 sudo
dnf install -y sudo || yum install -y sudo

# 授予 kuma 免密 sudo 权限（可选）
echo "kuma ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/kuma
chmod 440 /etc/sudoers.d/kuma
```
