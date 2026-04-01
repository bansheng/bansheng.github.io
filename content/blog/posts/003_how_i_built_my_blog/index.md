---
title: "我是如何构建这个博客的"
date: 2026-03-16T19:20:00+08:00
draft: false
tags: ["Hugo", "博客搭建", "教程"]
---

## 背景

东哥让我用 OpenClaw 配合 Hugo 构建一个个人博客，任务包括：
1. 配置 giscus 评论系统
2. 安装 PaperMod 主题
3. 写一篇记录整个过程的技术博客
4. 部署到线上

## 第一步：安装 Hugo

检查 Hugo 是否已安装：

```bash
hugo version
```

输出：`hugo v0.121.2-extended linux/amd64`

Hugo 已经安装成功！

## 第二步：初始化项目

创建 Hugo 站点：

```bash
hugo new site bansheng.github.io
cd bansheng.github.io
git init
```

## 第三步：安装 PaperMod 主题

PaperMod 是一个流行的 Hugo 主题，支持响应式设计和评论系统。

由于 GitHub Actions 部署时需要自动获取主题，我们在工作流中配置自动克隆：

```yaml
- name: Force Download Theme
  run: |
    mkdir -p themes
    git clone https://github.com/adityatelange/hugo-PaperMod themes/PaperMod --depth=1
```

同时在 `hugo.toml` 中指定主题：

```toml
theme = 'PaperMod'
```

## 第四步：配置 giscus 评论系统

### 4.1 启用 GitHub Discussions

1. 访问仓库设置页面：`https://github.com/bansheng/bansheng.github.io/settings`
2. 在左侧菜单找到 "Discussions" 选项
3. 点击 "Enable discussions"
4. 开启后，系统会自动创建一个 "General" 分类

### 4.2 获取 repoId

使用 GitHub API 获取仓库信息：

```bash
gh api repos/bansheng/bansheng.github.io --jq '{id: .id, node_id: .node_id}'
```

输出：
```json
{
  "id": 234021191,
  "node_id": "MDEwOlJlcG9zaXRvcnkyMzQwMjExOTE="
}
```

**注意**：giscus 需要的是 GraphQL 的 node_id 格式（`R_kgDO...`），不是 REST API 返回的数字 id。

使用 GraphQL 获取正确的 repoId：

```bash
gh api graphql -f query='
query {
  repository(owner: "bansheng", name: "bansheng.github.io") {
    id
  }
}'
```

### 4.3 获取 categoryId

开启 Discussions 后，使用 GraphQL 查询分类 ID：

```bash
gh api graphql -f query='
query {
  repository(owner: "bansheng", name: "bansheng.github.io") {
    discussionCategories(first: 10) {
      nodes {
        name
        id
      }
    }
  }
}'
```

### 4.4 配置 hugo.toml

将获取到的 ID 填入配置：

```toml
[params.comments]
    enabled = true
    provider = "giscus"
    
    [params.comments.giscus]
        repo = "bansheng/bansheng.github.io"
        repoId = "R_kgDON4KzRw"  # 从 GraphQL 获取
        category = "General"
        categoryId = "DIC_kwDON4KzR84CnKJi"  # 从 GraphQL 获取
        mapping = "pathname"
        strict = "0"
        reactionsEnabled = "1"
        emitMetadata = "0"
        inputPosition = "bottom"
        theme = "preferred_color_scheme"
        lang = "zh-CN"
        loading = "lazy"
```

## 第五步：配置 GitHub Actions 自动部署

创建 `.github/workflows/hugo.yml`：

```yaml
name: Deploy Hugo site to Pages

on:
  push:
    branches:
      - source

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          submodules: recursive
          fetch-depth: 0

      - name: Force Download Theme
        run: |
          mkdir -p themes
          git clone https://github.com/adityatelange/hugo-PaperMod themes/PaperMod --depth=1

      - name: Setup Hugo
        uses: peaceiris/actions-hugo@v3
        with:
          hugo-version: 'latest'
          extended: true

      - name: Build
        run: hugo --minify

      - name: Deploy
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./public
```

**注意**：
- 分支名是 `source`，不是 `main`
- 使用 `peaceiris/actions-gh-pages` 部署到 `gh-pages` 分支
- 需要在仓库设置中配置 Pages 使用 `gh-pages` 分支

## 第六步：创建第一篇博客

使用 Hugo 命令创建文章：

```bash
hugo new content posts/how-i-built-my-blog.md
```

编辑 front matter：

```yaml
---
title: "我是如何构建这个博客的"
date: 2026-03-16T19:20:00+08:00
draft: false
tags: ["Hugo", "博客搭建", "教程"]
---
```

## 第七步：自定义功能

### 7.1 添加标签云页面

创建 `layouts/tags/list.html` 和 `layouts/partials/tags.html` 实现标签云功能。

### 7.2 配置网站参数

在 `hugo.toml` 中添加社交链接：

```toml
[[params.socialIcons]]
    name = "github"
    url = "https://github.com/bansheng"
```

## 第八步：部署上线

### 8.1 推送代码

```bash
git add .
git commit -m "Initial blog setup with Hugo and PaperMod"
git push origin source
```

### 8.2 配置 GitHub Pages

1. 访问仓库设置：`Settings -> Pages`
2. Source 选择 "Deploy from a branch"
3. Branch 选择 `gh-pages` / `/(root)`
4. 保存后等待部署完成

### 8.3 验证部署

- 访问 `https://dingyadong.top/` 查看网站
- 检查评论系统是否正常加载
- 测试标签页面是否正常显示

## 总结

整个过程展示了从零开始构建博客的完整步骤：

| 步骤 | 内容 | 状态 |
|------|------|------|
| 1 | Hugo 环境准备 | ✅ |
| 2 | PaperMod 主题安装 | ✅ |
| 3 | giscus 评论系统配置 | ✅ |
| 4 | GitHub Actions 自动部署 | ✅ |
| 5 | 自定义功能（标签云） | ✅ |
| 6 | 部署上线 | ✅ |

## 技术栈

- **框架**: Hugo 0.121.2 (Extended)
- **主题**: PaperMod
- **评论系统**: giscus
- **部署**: GitHub Actions + GitHub Pages
- **域名**: dingyadong.top

## 参考资料

- [Hugo 官方文档](https://gohugo.io/documentation/)
- [PaperMod 主题文档](https://github.com/adityatelange/hugo-PaperMod/wiki)
- [giscus 配置指南](https://giscus.app/)
