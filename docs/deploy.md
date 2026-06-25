# 服务器部署备忘

> 仅供自己使用，不提交到公开 README。

## 服务器信息

| 项 | 值 |
|---|---|
| 别名 | `sjtu-a800`（经 `sjtu-chaos` 跳板） |
| 用户 | `gaojc` |
| 项目路径 | `/data/gaojc/projects/deep-research-agent` |
| LLM | DeepSeek API（`deepseek-chat` / `deepseek-reasoner`） |

## 部署步骤

### 1. 同步代码

```bash
# 本地 commit & push 到 GitHub 后，在服务器上：
ssh sjtu-a800
cd /data/gaojc/projects/deep-research-agent
git pull origin master
```

### 2. 检查 .env

```bash
grep -E 'TAVILY|LLM_API|LLM_MODEL|LLM_BASE' .env
```

### 3. 启动

```bash
# 先杀掉旧进程
pkill -f "uvicorn app.main" 2>/dev/null

# 前台（调试用）
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8080

# 后台（长期运行）
cd /data/gaojc/projects/deep-research-agent
nohup .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 > /tmp/dra.log 2>&1 &
```

### 4. 本地访问

```bash
# 本地终端先建隧道：
ssh -L 8080:127.0.0.1:8080 sjtu-a800

# 浏览器打开：
# http://localhost:8080
```

### 5. 查看日志

```bash
ssh sjtu-a800 'tail -50 /tmp/dra.log'
```

### 6. 运行真实测试

```bash
ssh sjtu-a800
cd /data/gaojc/projects/deep-research-agent
.venv/bin/python -m pytest tests/test_quick_research.py tests/test_workflow.py -v
```
