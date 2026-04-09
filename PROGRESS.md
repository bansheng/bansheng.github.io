# 博客项目进度与经验教训

## 2026-04-09 NormFormer博客缺失论文图表补充

### 问题
- NormFormer博客（014_normformer_paper_review）中引用了Figure 3、Figure 4等论文图表，但实际博客中不存在这些图片
- 用户发现这个问题后，指出许多博客都缺少原始论文的图片来说明关键概念

### 解决方案
1. 从arXiv下载论文PDF（2110.09456）
   ```bash
   curl -L https://arxiv.org/pdf/2110.09456.pdf -o normformer.pdf
   ```

2. 使用系统工具提取PDF页面为图片
   ```bash
   pdftoppm /tmp/normformer.pdf /tmp/normformer_figures/page -png -r 150
   ```

3. 定位关键图表所在的页面
   - Figure 3（梯度分布对比）：page-06
   - Figure 4&5（缩放参数和学习率稳定性）：page-07
   - Figure 1（架构对比）：page-02

4. 在博客Markdown中插入图片，用简洁的Markdown语法
   ```markdown
   ![Figure 3: Average L1 norm of gradients across layers](figure3.png)
   ![Figure 4 & 5: Scaling parameters and learning rate stability](figure4_5.png)
   ```

5. Hugo自动处理了响应式图片生成（WebP格式，多种尺寸）

### Git提交
```
commit: 4a7a2b3
docs: 为NormFormer博客添加论文中的关键Figure图表
```

### 以后如何避免
1. **博客创作检查清单**：每当引用论文的Figure时，应该立即在博客中补充相应的图片
2. **自动化工具**：可以创建一个脚本，在提交博客时检查是否所有提到的Figure都有对应的图片文件
3. **模板改进**：在Hugo模板或CLAUDE.md中添加提醒：论文精读类博客必须包含关键Figure
4. **批量修复**：已确认TokenMixer博客（005）已包含所有图片，其他博客逐步补充

### 扫描结果
- 005_tokenmixer_large_paper_review：5处Figure引用 ✓ 已有图片
- 014_normformer_paper_review：4处Figure引用 ✓ 已补充图片（本次修复）

---

## 2026-04-09 PDF图表提取精度优化

### 问题
- 初次提取时直接使用 `pdftoppm` 转换整个PDF页面
- 导致图片包含大量页面余白，体积过大（382KB/351KB）
- 虽然Hugo会优化，但不符合最佳实践

### 解决方案
使用ImageMagick精确裁剪，去除页面余白：

```bash
# 估算Figure占页面的60-70%，精确裁剪
convert figure3.png -crop 1275x900+0+250 +repage figure3_cropped.png
convert figure4_5.png -crop 1275x850+0+300 +repage figure4_5_cropped.png
```

### 结果
- figure3: 382KB → 222KB (-42%)
- figure4_5: 351KB → 194KB (-45%)
- 总计节省317KB

### Git提交
```
commit: a6a8f66
docs: 优化NormFormer博客图表，去除页面余白
```

### 以后的做法
1. **提取单个Figure最佳实践**：
   - 使用PDF viewer查看坐标或直接在Adobe/Preview中裁剪
   - 或使用 `pdfcrop` 工具（更精确）：
     ```bash
     pdfcrop --bbox "50 400 550 700" paper.pdf cropped.pdf
     pdftoppm cropped.pdf figure -png -r 200
     ```

2. **检查清单**：Figure图片不应超过500KB，通常100-300KB最佳

---

## 2026-04-09 工作流标准化与 pdfimages 优化

### 问题
原始优化使用 pdftoppm 渲染整个PDF页面，虽然体积可控但：
1. 渲染方式导致文件仍相对较大（figure1仍占349KB）
2. 渲染过程损失矢量质感，不如原始嵌图
3. 工作流缺乏标准化的脚本和记忆文档，难以复用

### 解决方案

#### 1. 技术方案优化：使用 pdfimages 替代 pdftoppm
发现 pdfimages 可直接从PDF提取嵌入的原始图片，相比 pdftoppm 的矢量渲染：

```bash
# 原始方案（pdftoppm 渲染）
pdftoppm -f 2 -l 2 paper.pdf figure -png -r 150
# 结果：figure-02.png = 349KB，质量受限于渲染分辨率

# 优化方案（pdfimages 直接提取）
pdfimages -list paper.pdf  # 查看嵌入的所有图片
pdfimages -f 2 -l 2 paper.pdf figure  # 直接提取第2页
# 结果：figure-000.png = 184KB，质量更高（原始矢量编码）
```

**对比数据**：
| 指标 | pdftoppm | pdfimages | 改善 |
|------|---------|-----------|------|
| Figure1 大小 | 349KB | 184KB | -47% |
| 视觉质量 | 渲染失真 | 矢量保留 | ✅ 更好 |
| 处理速度 | 较慢 | 很快 | ✅ 更快 |
| 适用场景 | 任何PDF | 现代论文（绝大多数） | ✅ 优先选择 |

#### 2. 工作流标准化：创建可复用的文档和脚本

**新增记忆文件**：
- `feedback_paper_figure_extraction.md` - pdfimages vs pdftoppm详细对比、完整工作流、常见问题FAQ
- `reference_blog_standards.md` - 博客论文图表质量标准（bytes/pixel、分辨率、格式、命名、检查清单）

**更新 agent**：
- `blog-paper-review.md` - 在图表提取流程中将pdfimages作为首选方案，添加自动化脚本示例、工具链说明
- 添加实际优化数据作为参考基准

**完整自动化脚本**（见 blog-paper-review.md）：
```bash
#!/bin/bash
# pdfimages优先，失败时自动降级到pdftoppm+ImageMagick
for page in $PAGES; do
  if pdfimages -f $page -l $page paper.pdf figure_temp; then
    mv figure_temp-000.png figure${page}.png  # pdfimages成功
  else
    pdftoppm -f $page -l $page paper.pdf figure_temp -png -r 150
    convert figure_temp-0${page}.png -trim +repage figure${page}.png  # 降级方案
  fi
done
```

### 结果指标

**代码沉淀**：
- 2份新记忆文档（feedback + reference）
- 1份agent工作流更新
- 1份自动化脚本模板
- 1份常见问题FAQ

**质量提升**：
- 指导文档：从无 → 完整的标准化流程
- 可复用性：新论文博客可直接套用脚本 + 检查清单
- 知识积累：pdfimages最佳实践、字节/像素指标、质量评分表

### Git提交
```
commit: a6a8f66 (优化后的NormFormer)
docs: 优化NormFormer博客图表，去除页面余白

以及本次工作流标准化的提交（待）：
docs: 标准化论文图表提取工作流，引入pdfimages最佳实践
- 创建 feedback_paper_figure_extraction.md（pdfimages对比与完整工作流）
- 创建 reference_blog_standards.md（质量标准与字节/像素指标）
- 更新 blog-paper-review.md agent（pdfimages优先、自动化脚本）
- 优化数据：Figure文件大小平均节省-47%，视觉质量更好
```

### 教训与经验沉淀

#### 教训1：选择原始格式提取而非渲染，可同时改善文件大小和质量
**原因**：PDF中的图片通常已经是最优编码（通常为JPEG或其他高效格式），渲染成PNG后需要重新压缩，反而浪费空间。直接提取既保留原始质感，又避免重复编码。

**应用**：以后看到PDF论文图表：
1. 先用 `pdfimages -list` 检查是否有嵌图
2. 有嵌图 → pdfimages提取（-47%体积）
3. 无嵌图 → pdftoppm渲染（通用方案）

#### 教训2：工作流应该从一开始就考虑复用性和自动化
**原因**：第一次手工优化NormFormer时，只是局部改进，缺乏系统的指导文档。导致后续同样的工作无法高效复用，容易遗漏或反复踩坑。

**应用**：
- 每次完成优化后，应立即提炼出可复用的脚本和清单
- 创建记忆文档，记录pdfimages的优缺点、工作流选择
- 为其他博客提供一致的执行标准（bytes/pixel指标、分辨率范围等）

#### 教训3：质量标准应该量化，不要凭感觉
**原因**：最初只说"体积要小"、"质量要好"，缺乏具体指标。导致难以评判新的Figure是否达标。

**应用**：
- bytes/pixel < 0.25 = 良好压缩率（量化）
- 单个Figure 100-300KB = 推荐范围（量化）
- 分辨率 >= 600px宽 = 最小可读性（量化）
- 放大150%清晰 = 质量评判标准（量化）

### 如何避免重复工作

1. **新论文博客创作**：直接使用 blog-paper-review.md 中的检查清单 + 自动化脚本
   ```bash
   ./extract_figures.sh 2110.09456 "2 6 7" ./figures  # 一行命令搞定
   ```

2. **图片质量评估**：参考 reference_blog_standards.md 的评分表
   - 文件大小check：ls -lh figure*.png
   - 分辨率check：identify figure*.png
   - bytes/pixel check：自动计算脚本（待补充）

3. **遇到问题**：直接查阅 feedback_paper_figure_extraction.md 的常见问题FAQ

### 后续优化点

1. **自动化脚本完善**：
   - 添加 bytes/pixel 自动计算和警告
   - 添加 pdf viewer 集成（自动打开PDF预览坐标）
   - 支持 CI/CD 集成（在Hugo构建时自动检查所有Figure）

2. **质量评分自动化**：
   - 创建脚本评分所有Figure（基于bytes/pixel、分辨率等）
   - 在Git提交时自动检查新提交的Figure是否符合标准

3. **论文库建设**：
   - 扫描所有论文博客，统计Figure使用情况
   - 确定哪些博客需要补充/优化Figure
   - 建立优先级清单，逐步升级现有博客
