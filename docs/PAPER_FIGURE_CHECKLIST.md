# 论文图表提取 - 检查清单和质量标准

**最后更新**: 2026-04-09

---

## 目录

1. [提取前检查清单](#提取前检查清单)
2. [提取过程检查清单](#提取过程检查清单)
3. [验收标准](#验收标准)
4. [集成到博客的检查清单](#集成到博客的检查清单)
5. [常见不符合项及修复](#常见不符合项及修复)

---

## 提取前检查清单

### 一、准备工作

- [ ] 获取论文 PDF 文件
  - [ ] 检查文件完整性（不损坏）
  - [ ] 确认文件大小合理（> 100KB，< 200MB）
  - [ ] 文件编码正确（不是压缩格式如 .rar/.zip）

- [ ] 安装必需工具
  - [ ] `pdfimages` (poppler)
    ```bash
    pdfimages --version
    # 输出: pdfimages version X.XX.X
    ```
  - [ ] `pdftoppm` (poppler)
    ```bash
    pdftoppm --version
    # 输出: pdftoppm version X.XX.X
    ```
  - [ ] `convert` (ImageMagick)
    ```bash
    convert --version
    # 输出: Version: ImageMagick X.X.X
    ```
  - [ ] `identify` (ImageMagick)
    ```bash
    identify --version
    # 输出: Version: ImageMagick X.X.X
    ```
  - [ ] `bc` (计算器)
    ```bash
    bc --version
    # 输出: GNU bc X.XX
    ```

### 二、论文审阅

- [ ] 打开 PDF，查看所有 Figure
  - [ ] 统计 Figure 总数
  - [ ] 记录每个 Figure 的页码
  - [ ] 检查 Figure 质量（清晰度、大小）
  - [ ] 标记需要提取的 Figure

- [ ] 确定提取策略
  - [ ] 提取全部 Figure 还是部分？
  - [ ] 是否需要裁剪去除余白？
  - [ ] 是否需要调整分辨率？

**示例**（NormFormer 论文）:
```
总共 4 个 Figure：
□ Figure 1: page 2, 标题 "架构对比", 清晰
□ Figure 2: page 4, 标题 "学习曲线", 模糊 (可能不提取)
□ Figure 3: page 6, 标题 "梯度分布", 清晰
□ Figure 4&5: page 7, 标题 "参数设置", 清晰

需提取: Figure 1, 3, 4&5 (页码: 2, 6, 7)
```

### 三、环境检查

- [ ] 检查硬盘空间
  - [ ] 至少 1GB 可用空间（用于临时文件）
  - [ ] 输出目录有写权限
  ```bash
  df -h
  touch /output/dir/test && rm /output/dir/test
  ```

- [ ] 检查网络（如需下载 PDF）
  ```bash
  curl -I https://arxiv.org/pdf/XXXX.XXXXX.pdf
  # 应该返回 200 OK
  ```

---

## 提取过程检查清单

### 步骤 1: 下载 PDF

- [ ] 使用正确的 URL
  ```bash
  # arXiv 论文
  curl -L https://arxiv.org/pdf/2110.09456.pdf -o /tmp/normformer.pdf
  
  # 检查文件
  file /tmp/normformer.pdf
  # 输出: PDF document, version 1.5
  ```

- [ ] 验证下载完整性
  ```bash
  # 检查文件大小
  ls -lh /tmp/normformer.pdf
  # 应该是几 MB，不应该太小
  
  # 检查是否可读
  pdfinfo /tmp/normformer.pdf
  ```

### 步骤 2: 运行提取脚本

- [ ] 参数正确
  ```bash
  ./scripts/extract_paper_figures.sh \
    --pdf /tmp/normformer.pdf \
    --pages "2,6,7" \
    --output-dir ./content/blog/posts/014_normformer_paper_review
  ```

- [ ] 监控执行过程
  - [ ] 检查日志输出是否有错误
  - [ ] 注意是否有 WARNING（表示降级）
  - [ ] 检查最终统计 (成功 3/3 或其他)

- [ ] 验证输出
  ```bash
  # 检查输出文件
  ls -lh content/blog/posts/014_normformer_paper_review/
  
  # 应该包含:
  # - figure_1.png
  # - figure_2.png (如果提取了 2 个)
  # - figure_3.png
  # - figures_quality_report.md
  ```

### 步骤 3: 检查质量报告

- [ ] 打开质量报告
  ```bash
  cat content/blog/posts/014_normformer_paper_review/figures_quality_report.md
  ```

- [ ] 逐个检查每个 Figure
  - [ ] 尺寸是否满足要求（≥ 600×400px）
  - [ ] 文件大小是否合理（< 500KB）
  - [ ] bytes/pixel 是否良好（< 0.5）
  - [ ] 质量评级是否 ≥ ⭐⭐⭐

**示例报告**:
```markdown
# 论文图表提取质量报告

### Figure 1: figure_1.png
- **尺寸**: 1275×900px ✓
- **文件大小**: 184.0KB ✓
- **bytes/pixel**: 0.1593 ✓
- **质量评级**: ⭐⭐⭐⭐⭐ 优秀 ✓
- **状态**: ✓ 通过

### Figure 2: figure_2.png
- **尺寸**: 1275×1500px ✓
- **文件大小**: 222.0KB ✓
- **bytes/pixel**: 0.1159 ✓
- **质量评级**: ⭐⭐⭐⭐⭐ 优秀 ✓
- **状态**: ✓ 通过

### Figure 3: figure_3.png
- **尺寸**: 1275×1100px ✓
- **文件大小**: 194.0KB ✓
- **bytes/pixel**: 0.1374 ✓
- **质量评级**: ⭐⭐⭐⭐⭐ 优秀 ✓
- **状态**: ✓ 通过
```

---

## 验收标准

### 一、分辨率标准

| 等级 | 宽度 | 高度 | 用途 | 通过 |
|------|------|------|------|------|
| **最低** | ≥ 600px | ≥ 400px | 基本可读 | ✓ |
| **推荐** | 800-1200px | 600-900px | 博客标准 | ✓✓ |
| **高清** | ≥ 1200px | ≥ 800px | 高质量展示 | ✓✓✓ |

**检查**:
```bash
# 查看分辨率
identify content/blog/posts/014_normformer_paper_review/figure_*.png

# 输出示例:
# figure_1.png PNG 1275x900 1275x900+0+0 8-bit sRGB 184KB
```

**标准**:
- [ ] 所有 Figure 宽度 ≥ 600px
- [ ] 所有 Figure 高度 ≥ 400px
- [ ] 推荐至少一半 Figure 达到推荐级别

### 二、文件大小标准

| 分类 | 单个 Figure | 总计 | 说明 |
|------|-----------|------|------|
| **严格** | ≤ 150KB | ≤ 500KB | 移动优先 |
| **标准** | ≤ 300KB | ≤ 1MB | 博客标准 |
| **宽松** | ≤ 500KB | ≤ 2MB | 高质量文档 |

**检查**:
```bash
# 查看文件大小
ls -lh content/blog/posts/014_normformer_paper_review/figure_*.png

# 计算总大小
du -sh content/blog/posts/014_normformer_paper_review/
```

**标准**:
- [ ] 单个 Figure ≤ 500KB（标准）
- [ ] 所有 Figure 总计 ≤ 1.5MB
- [ ] 平均 Figure 大小 ≤ 300KB

### 三、压缩效率标准 (bytes/pixel)

| 等级 | bytes/pixel | 评级 | 压缩效率 | 通过 |
|------|------------|------|---------|------|
| **优秀** | < 0.15 | ⭐⭐⭐⭐⭐ | 极好 | ✓✓✓ |
| **很好** | 0.15-0.25 | ⭐⭐⭐⭐ | 很好 | ✓✓ |
| **可接受** | 0.25-0.5 | ⭐⭐⭐ | 可接受 | ✓ |
| **一般** | 0.5-1.0 | ⭐⭐ | 需要优化 | ⚠ |
| **差** | > 1.0 | ⭐ | 必须优化 | ❌ |

**计算**:
```bash
# 手工计算 Figure 1
# 文件大小: 184,320 字节
# 宽度: 1275 像素
# 高度: 900 像素
# bytes/pixel = 184320 / (1275 × 900) = 0.1606 ✓ 优秀
```

**标准**:
- [ ] 所有 Figure 的 bytes/pixel ≤ 0.5（可接受）
- [ ] 推荐至少 50% 的 Figure ≤ 0.25（很好）
- [ ] 优秀目标: 所有 Figure ≤ 0.25（很好及以上）

### 四、格式和颜色空间标准

**必需**:
- [ ] 所有 Figure 格式统一（全部 PNG 或全部 JPG）
- [ ] 推荐使用 PNG（更好的无损压缩）
- [ ] 颜色空间为 RGB 或 sRGB
- [ ] 位深为 8-bit 或更高

**检查**:
```bash
# 检查所有格式和颜色空间
identify -verbose content/blog/posts/014_normformer_paper_review/figure_*.png | grep -E "Format|Colorspace|Depth"

# 示例输出应该是:
# Format: PNG
# Colorspace: sRGB
# Depth: 8-bit
```

---

## 集成到博客的检查清单

### 步骤 1: 更新 Markdown 文件

- [ ] 打开博客 Markdown 文件
  ```bash
  vim content/blog/posts/014_normformer_paper_review/index.md
  ```

- [ ] 找到引用 Figure 的位置
  ```markdown
  # 架构设计
  根据论文描述，NormFormer 架构如下...
  
  ![Figure 1: 架构对比](figure_1.png)
  ```

- [ ] 验证引用正确
  - [ ] 使用相对路径（不是绝对路径）
  - [ ] 文件名与生成的文件匹配
  - [ ] 图片说明清晰准确

**示例 Markdown**:
```markdown
## 架构设计

![Figure 1: NormFormer 与 Transformer 架构对比](figure_1.png)

NormFormer 的核心创新是引入了 ...

## 实验结果

### 梯度分布分析

![Figure 3: 不同层级的平均梯度 L1 范数](figure_3.png)

从图 3 可以看出，...

### 超参数敏感性

![Figure 4 & 5: 缩放参数和学习率稳定性](figure_4_5.png)

缩放参数 $\alpha$ 和 $\beta$ 的设置对...
```

### 步骤 2: 本地预览

- [ ] 启动 Hugo 服务
  ```bash
  cd /Users/bytedance/03_personal/bansheng.github.io
  hugo server
  ```

- [ ] 访问本地地址
  ```
  http://localhost:1313/
  ```

- [ ] 检查图片显示
  - [ ] 图片加载成功（没有 404）
  - [ ] 大小合适（没有拉伸或压缩变形）
  - [ ] 对齐正确（居中或靠左）
  - [ ] 说明文本清晰

- [ ] 检查响应式显示
  - [ ] 在桌面浏览器查看（全宽）
  - [ ] 在平板模式查看（50% 缩放）
  - [ ] 在手机模式查看（100% 宽度）
  - [ ] 确保在所有尺寸下都清晰可读

### 步骤 3: Git 提交前检查

- [ ] 检查所有修改的文件
  ```bash
  git status
  
  # 应该显示:
  # modified:   content/blog/posts/014_normformer_paper_review/index.md
  # new file:   content/blog/posts/014_normformer_paper_review/figure_1.png
  # new file:   content/blog/posts/014_normformer_paper_review/figure_3.png
  # new file:   content/blog/posts/014_normformer_paper_review/figure_4_5.png
  ```

- [ ] 检查文件大小
  ```bash
  ls -lh content/blog/posts/014_normformer_paper_review/
  
  # 图片文件应该是 100-300KB
  # Markdown 文件应该是 20-50KB
  ```

- [ ] 检查 git diff
  ```bash
  git diff content/blog/posts/014_normformer_paper_review/index.md
  
  # 应该只增加 Figure 引用，没有其他修改
  ```

- [ ] 确认没有遗漏文件
  ```bash
  git add -A
  git status
  
  # 确保所有 Figure 和 Markdown 都已 staged
  ```

### 步骤 4: 提交和推送

- [ ] 编写清晰的 commit 信息
  ```bash
  git commit -m "docs: 为NormFormer博客添加论文图表（Figure 1,3,4,5）

  - 使用 pdfimages 提取原始嵌图，保持高质量
  - Figure 1: 1275×900px, 184KB, 0.16 bytes/pixel ⭐⭐⭐⭐⭐
  - Figure 3: 1275×1500px, 222KB, 0.12 bytes/pixel ⭐⭐⭐⭐⭐
  - Figure 4&5: 1275×1100px, 194KB, 0.14 bytes/pixel ⭐⭐⭐⭐⭐
  
  所有 Figure 均通过质量验证。"
  ```

- [ ] 推送到远程
  ```bash
  git push origin source
  # 或如果设置了 upstream
  git push
  ```

- [ ] 确认 CI 通过
  - [ ] GitHub Actions 构建成功
  - [ ] 所有检查通过
  - [ ] 没有警告或错误

### 步骤 5: 发布确认

- [ ] 检查网站是否发布成功
  ```bash
  # 等待 GitHub Pages 构建（通常 1-2 分钟）
  # 访问 https://bansheng.github.io
  ```

- [ ] 在发布版本上验证
  - [ ] 图片加载正确
  - [ ] 显示质量满意
  - [ ] 没有链接错误

- [ ] 更新进度记录
  ```bash
  # 在 PROGRESS.md 中添加记录
  echo "2026-04-09: NormFormer 博客完成图表补充" >> PROGRESS.md
  ```

---

## 常见不符合项及修复

### 不符合项 1: 分辨率过小 (< 600px)

**症状**:
```
❌ 宽度过小: 480px < 600px
```

**原因**:
- pdfimages 提取的是 PDF 中缩小的嵌图
- pdftoppm 分辨率设置过低

**修复方案**:

**方案 A**: 使用更高的 pdftoppm DPI
```bash
# 编辑脚本，找到这一行:
# pdftoppm -f $page -l $page -png -r 150 "$pdf" "$output_prefix"

# 改为:
pdftoppm -f $page -l $page -png -r 200 "$pdf" "$output_prefix"
# 或更高:
# pdftoppm -f $page -l $page -png -r 300 "$pdf" "$output_prefix"
```

**方案 B**: 从原始 PDF 扩大
```bash
# 使用 ImageMagick 放大
convert figure_1.png -resize 150% figure_1_enlarged.png

# 或用高质量算法
convert figure_1.png -resize 150% -quality 85 -filter Lanczos figure_1_enlarged.png
```

**方案 C**: 检查源 PDF 质量
```bash
# 查看 PDF 中的嵌图信息
pdfimages -list paper.pdf

# 如果嵌图本身就小，无法修复
# 建议改用 pdftoppm
```

### 不符合项 2: 文件过大 (> 500KB)

**症状**:
```
❌ 文件过大: 624.5KB > 500KB 限制
```

**原因**:
- pdftoppm 渲染分辨率过高
- 图片包含大量页面背景
- 压缩率不佳

**修复方案**:

**方案 A**: 降低 pdftoppm 分辨率
```bash
# 编辑脚本，从 -r 200 改为 -r 150
pdftoppm -f $page -l $page -png -r 150 "$pdf" "$output_prefix"
```

**方案 B**: 压缩 PNG
```bash
# 使用 ImageMagick 压缩
convert figure_1.png -strip -quality 85 figure_1_compressed.png

# 或使用 pngquant 高级压缩
pngquant 256 figure_1.png -o figure_1_optimized.png
```

**方案 C**: 裁剪去除余白
```bash
# 自动检测并裁剪
convert figure_1.png -trim +repage figure_1_trimmed.png

# 或手工指定裁剪区域 (根据 PDF 查看器确定)
convert figure_1.png -crop 1200x800+0+100 +repage figure_1_cropped.png
```

**方案 D**: 调整最大大小限制（不推荐）
```bash
# 只在有充分理由时调整
./scripts/extract_paper_figures.sh \
  --pdf paper.pdf \
  --pages "2,6" \
  --output-dir ./output \
  --max-size 800000  # 从 500KB 改为 800KB
```

### 不符合项 3: 压缩效率差 (bytes/pixel > 0.5)

**症状**:
```
⚠ 压缩率: 0.67 bytes/pixel (>0.5 阈值)
```

**原因**:
- pdftoppm 渲染的图表包含复杂内容
- PNG 压缩不适合这类内容
- 原始 PDF 有低质量嵌图

**修复方案**:

**方案 A**: 使用 JPG 替代 PNG
```bash
# 如果允许有损压缩
convert figure_1.png -quality 85 figure_1.jpg

# bytes/pixel 会降低到 0.15-0.25
```

**方案 B**: 改用 pdfimages（如果尚未尝试）
```bash
# 检查是否有更好的嵌图可提取
pdfimages -list paper.pdf

# 如果有，重新用 pdfimages 提取
pdfimages -f 2 -l 2 paper.pdf figure_temp
```

**方案 C**: 调整质量标准
```bash
# 如果 bytes/pixel 在 0.5-1.0 范围且图片质量可接受
# 可以调整阈值
./scripts/extract_paper_figures.sh \
  --pdf paper.pdf \
  --pages "2" \
  --output-dir ./output \
  --quality-threshold 0.7  # 从 0.5 改为 0.7
```

### 不符合项 4: 图片模糊或失真

**症状**:
```
质量评级: ⭐⭐ 一般，图片看起来模糊
```

**原因**:
- 源 PDF 嵌图质量低
- pdftoppm 渲染参数不合适
- 文件压缩过度

**修复方案**:

**方案 A**: 检查源 PDF
```bash
# 用 Preview 或 Adobe 查看原始 PDF
open paper.pdf

# 如果源 PDF 中本身模糊，无法修复
# 建议使用原论文的高清版本
```

**方案 B**: 增加 pdftoppm 分辨率
```bash
# 从 -r 150 改为 -r 300（会增加文件大小）
pdftoppm -f 2 -l 2 -png -r 300 paper.pdf figure

# 然后可能需要压缩
convert figure-002.png -quality 90 figure_final.png
```

**方案 C**: 使用高质量重采样
```bash
# 重新缩放时使用好的算法
convert figure_1.png -resize 120% -filter Lanczos figure_1_hq.png
```

### 不符合项 5: 颜色空间错误 (CMYK / Indexed)

**症状**:
```
Warning: 颜色空间为 CMYK，可能在网页显示时不正确
```

**原因**:
- pdfimages 提取的原始图片是 CMYK（通常用于印刷）
- 需要转换为 RGB 用于网页显示

**修复方案**:

```bash
# 转换 CMYK 到 RGB
convert figure_1.png -colorspace RGB figure_1_rgb.png

# 或使用配置文件转换
convert figure_1.png -profile /usr/share/color/icc/srgb.icc figure_1_srgb.png

# 验证转换
identify figure_1_rgb.png
# 应该显示 "Colorspace: sRGB"
```

### 不符合项 6: 提取失败

**症状**:
```
ERROR: 页面 2 提取失败，跳过
```

**原因**:
- PDF 文件损坏
- 页码超出范围
- 工具未正确安装

**修复方案**:

**步骤 1**: 验证 PDF 完整性
```bash
# 检查 PDF
pdfinfo paper.pdf

# 如果输出有"Encrypted"或错误，PDF 可能有问题
# 尝试修复
gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile=paper_fixed.pdf paper.pdf
```

**步骤 2**: 验证页码
```bash
# 获取 PDF 总页数
pdfinfo paper.pdf | grep Pages

# 确保指定的页码在范围内
# 如果总页数是 11，不能指定页 15
```

**步骤 3**: 重新安装工具
```bash
# macOS
brew reinstall poppler imagemagick

# 验证
pdfimages --version
convert --version
```

**步骤 4**: 使用详细模式调试
```bash
./scripts/extract_paper_figures.sh \
  --pdf paper.pdf \
  --pages "2" \
  --output-dir ./output \
  --verbose

# 查看详细输出找出原因
```

---

## 快速参考

### 检查清单速查表

```bash
# ============ 提取前 ============
□ 下载 PDF: curl -L URL -o /tmp/paper.pdf
□ 验证工具: pdfimages --version && pdftoppm --version && convert --version
□ 审阅 PDF: open /tmp/paper.pdf # 找出 Figure 页码
□ 检查空间: df -h

# ============ 提取 ============
□ 运行脚本: ./scripts/extract_paper_figures.sh ...
□ 监控输出: 检查是否有 ERROR
□ 验证输出: ls -lh output/

# ============ 验收 ============
□ 查看报告: cat output/figures_quality_report.md
□ 分辨率: identify output/figure_*.png (应该 ≥ 600×400px)
□ 文件大小: du -sh output/ (应该 ≤ 1.5MB)
□ bytes/pixel: (应该 ≤ 0.5)

# ============ 集成 ============
□ 编辑 Markdown: vim content/blog/posts/xxx/index.md
□ 本地预览: hugo server
□ 检查显示: http://localhost:1313
□ 提交: git add -A && git commit -m "..."
□ 推送: git push

# ============ 验收标准 ============
分辨率:     ✓ 宽度 ≥ 600px,  高度 ≥ 400px
文件大小:    ✓ 单个 ≤ 500KB,   总计 ≤ 1.5MB
压缩效率:    ✓ bytes/pixel ≤ 0.5
格式:       ✓ PNG 或 JPG,      RGB 颜色空间
完整性:     ✓ 所有 Figure 都提取且验证通过
```

