# Giscus 配置指南

## 第一步：开启 GitHub Discussions

1. 访问仓库设置页面：https://github.com/bansheng/bansheng.github.io/settings
2. 在左侧菜单找到 "Discussions" 选项
3. 点击 "Enable discussions"
4. 开启后，点击 "New category" 创建一个分类
   - Name: `General`
   - Description: `General discussion`
   - Visibility: `Public`

## 第二步：获取 Category ID

开启 Discussions 后，访问：
https://github.com/bansheng/bansheng.github.io/discussions/new

然后在浏览器控制台（F12 -> Console）运行以下代码获取 categoryId：

```javascript
fetch('https://api.github.com/repos/bansheng/bansheng.github.io/discussions_categories')
  .then(r => r.json())
  .then(data => {
    const categoryId = data.find(c => c.name === 'General')?.id;
    console.log('Category ID:', categoryId);
    alert('Category ID: ' + categoryId);
  });
```

## 第三步：配置 Giscus

将获取到的 categoryId 填入 hugo.toml 文件：

```toml
[params.comments.giscus]
    repo = "bansheng/bansheng.github.io"
    repoId = "R_kgDONxxxxx"  # 从 API 获取
    category = "General"
    categoryId = "DIC_kwDONxxxxx"  # 从上面获取
```

repoId 可以通过以下方式获取：

```javascript
fetch('https://api.github.com/repos/bansheng/bansheng.github.io')
  .then(r => r.json())
  .then(data => {
    console.log('Repo ID:', data.id);
    alert('Repo ID: ' + data.id);
  });
```

## 完整配置示例

```toml
[params.comments]
    enabled = true
    provider = "giscus"

    [params.comments.giscus]
        repo = "bansheng/bansheng.github.io"
        repoId = "R_kgDONxxxxx"
        category = "General"
        categoryId = "DIC_kwDONxxxxx"
        mapping = "pathname"
        strict = "0"
        reactionsEnabled = "1"
        emitMetadata = "0"
        inputPosition = "bottom"
        theme = "preferred_color_scheme"
        lang = "zh-CN"
        loading = "lazy"
```
