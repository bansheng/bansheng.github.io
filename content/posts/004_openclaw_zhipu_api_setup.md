---
title: "火山引擎 OpenClaw 配置智谱 API：绕过 Coding Plan 使用免费 API Key"
date: 2026-03-17T10:00:00+08:00
draft: false
tags: ["OpenClaw", "火山引擎", "智谱AI", "API", "教程"]
---

## 背景

火山引擎的 OpenClaw 是一个强大的 AI 编程助手，但默认情况下需要使用字节提供的 Coding Plan 进行计费。对于个人开发者或小型团队来说，如果能使用智谱 AI 提供的免费 API Key，可以大大降低成本。

本文将详细介绍如何在火山引擎 OpenClaw 中配置智谱 AI 的 API Key，实现自定义模型接入。

## 步骤

### 1. 注册智谱 AI 账号

访问 [智谱 AI 开放平台](https://open.bigmodel.cn/) 注册账号：

1. 使用手机号或邮箱注册
2. 完成实名认证（个人认证即可）
3. 进入控制台获取 API Key

### 2. 获取智谱 API Key

登录后进入 [API Keys 管理页面](https://open.bigmodel.cn/usercenter/apikeys)：

1. 点击 "添加新的 API Key"
2. 为 Key 命名（如 "OpenClaw-Config"）
3. 复制生成的 API Key（格式为 `sk-xxxxxxxxxxxxxxxx`）

> **注意**：新注册用户通常有免费的额度，足够个人开发使用。

### 3. 确认火山引擎 OpenClaw 已安装并配置好了飞书机器人

此时发送消息，飞书机器人报错提示并不存在 API Key。

我们直接修改 `~/.openclaw/openclaw.json`：

```json
"auth": {
  "profiles": {
    "zai:default": {
      "provider": "zai",
      "mode": "api_key"
    }
  }
},
"models": {
  "mode": "merge",
  "providers": {
    "zai": {
      "baseUrl": "https://open.bigmodel.cn/api/paas/v4",
      "api": "openai-completions",
      "models": [
        {
          "id": "glm-5",
          "name": "GLM-5",
          "reasoning": true,
          "input": [
            "text"
          ],
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          },
          "contextWindow": 204800,
          "maxTokens": 131072
        },
        {
          "id": "glm-4.7",
          "name": "GLM-4.7",
          "reasoning": true,
          "input": [
            "text"
          ],
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          },
          "contextWindow": 204800,
          "maxTokens": 131072
        },
        {
          "id": "glm-4.7-flash",
          "name": "GLM-4.7 Flash",
          "reasoning": true,
          "input": [
            "text"
          ],
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          },
          "contextWindow": 204800,
          "maxTokens": 131072
        },
        {
          "id": "glm-4.7-flashx",
          "name": "GLM-4.7 FlashX",
          "reasoning": true,
          "input": [
            "text"
          ],
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          },
          "contextWindow": 204800,
          "maxTokens": 131072
        }
      ]
    }
  }
},
"agents": {
  "defaults": {
    "models": {
      "zai/glm-4.7-flash": {
        "alias": "GLM"
      }
    },
    "workspace": "/root/.openclaw/workspace",
    "compaction": {
      "mode": "safeguard"
    },
    "maxConcurrent": 4,
    "subagents": {
      "maxConcurrent": 8
    }
  },
  "list": [
    {
      "id": "main",
      "name": "main",
      "workspace": "/root/.openclaw/workspace",
      "agentDir": "/root/.openclaw/agents/main/agent",
      "model": "zai/glm-4.7-flash"
    }
  ]
}
```

修改里面 `auth`、`models`、`agents` 的配置项，直接写死模型提供商和模型名。

接着配置 API Key：

```bash
vi ~/.openclaw/auth-profiles.json
```

写入以下内容（替换为你的 API Key）：

```json
{
  "zai": {
    "apiKey": "sk-xxxxxxxxxxxxxxxx",
    "baseURL": "https://open.bigmodel.cn/api/paas/v4/"
  },
  "openai": {
    "apiKey": "sk-xxxxxxxxxxxxxxxx",
    "baseURL": "https://open.bigmodel.cn/api/paas/v4/"
  },
  "anthropic": {
    "apiKey": "dummy-key"
  }
}
```

保存配置后退出编辑器。

### 4. 重启 OpenClaw Gateway

```bash
openclaw gateway restart
```

### 5. 飞书发送消息切换模型

在前端发送以下命令切换模型：

```
/model zai/glm-4.7-flash
```

切换成功后，就可以使用你配置好的智谱 API Key 来调用 OpenClaw 了。

## 常见问题

### Q1: 配置后无法连接

**可能原因**：

- API Key 错误或已过期
- Base URL 拼写错误
- 网络问题

**解决方法**：

1. 在智谱控制台确认 API Key 状态
2. 检查 URL 是否为 `https://open.bigmodel.cn/api/paas/v4/`
3. 尝试在浏览器访问智谱官网确认网络正常

### Q2: 响应速度慢

**优化建议**：

- 切换到 `glm-4.7-flash` 模型
- 减少 `max_tokens` 值
- 检查网络连接质量

### Q3: 代码质量不如预期

**改进方法**：

- 使用 `glm-5` 旗舰模型
- 在提示词中提供更详细的上下文

### Q4: 免费额度用完了怎么办

**解决方案**：

1. 智谱新用户通常有充足的免费额度
2. 可以关注智谱官方活动获取额外额度
3. 如果用量大，考虑购买付费套餐（比 Coding Plan 便宜）

## 与 Coding Plan 的对比

| 特性 | 智谱 API | Coding Plan |
|------|----------|-------------|
| 成本 | 免费额度 + 低价付费 | 按量计费 |
| 模型选择 | 多种模型可选 | 固定模型 |
| 自定义程度 | 高 | 低 |
| 稳定性 | 依赖智谱服务 | 火山引擎官方支持 |
| 适用场景 | 个人/小团队 | 企业/大团队 |

## 安全建议

1. **保护 API Key**: 不要将 API Key 提交到代码仓库
2. **使用环境变量**: 将 Key 存储在环境变量中
3. **定期轮换**: 定期更换 API Key
4. **监控用量**: 在智谱控制台监控 API 调用情况

## 总结

通过配置智谱 AI 的 API Key，我们可以在火山引擎 OpenClaw 中实现：

- ✅ 零成本或低成本使用 AI 编程助手
- ✅ 灵活选择不同能力的模型
- ✅ 自定义模型参数优化体验
- ✅ 避免被锁定在单一供应商

这种配置方式特别适合：

- 个人开发者学习使用
- 小型项目快速开发
- 对成本敏感的场景

## 参考资料

- [智谱 AI 开放平台](https://open.bigmodel.cn/)
- [智谱 API 文档](https://open.bigmodel.cn/dev/api)
- [OpenClaw 官方文档](https://www.volcengine.com/docs/)

---

*写于 2026年3月17日*
