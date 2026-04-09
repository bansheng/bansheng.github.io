# 论文图表提取工作流 - 使用指南

**最后更新**: 2026-04-09  
**版本**: 1.0

---

## 目录

1. [快速开始](#快速开始)
2. [工具对比（Bash vs Python）](#工具对比bash-vs-python)
3. [详细使用说明](#详细使用说明)
4. [与 Hugo 工作流集成](#与-hugo-工作流集成)
5. [批量处理](#批量处理)
6. [常见问题 FAQ](#常见问题-faq)
7. [故障排查](#故障排查)

---

## 快速开始

### 前置要求

```bash
# macOS
brew install imagemagick poppler

# Ubuntu/Debian
sudo apt-get install imagemagick poppler-utils

# CentOS/RHEL
sudo yum install ImageMagick poppler-utils
```

### 一键提取 (Bash 脚本)

```bash
# 从 NormFormer 论文提取第2、6、7页
./scripts/extract_paper_figures.sh \
  --pdf /tmp/normformer.pdf \
  --pages "2,6,7" \
  --output-dir ./content/blog/posts/014_normformer_paper_review
```

**预期输出**:
```
[INFO] 2026-04-09 14:30:00 ==========================================
[INFO] 2026-04-09 14:30:00 论文 PDF 图表提取工具
[INFO] 2026-04-09 14:30:00 ==========================================

[✓] 2026-04-09 14:30:00 所有必需工具已就绪
[INFO] 2026-04-09 14:30:00 输出目录: ./content/blog/posts/014_normformer_paper_review
[INFO] 2026-04-09 14:30:00 PDF 信息: /tmp/normformer.pdf
[INFO] 2026-04-09 14:30:00   - 大小: 1.42 MB
[INFO] 2026-04-09 14:30:00   - 页数: 11
[INFO] 2026-04-09 14:30:00 待提取页码: 2,6,7 (共 3 个)

[INFO] 2026-04-09 14:30:01 处理页面 #2 (Figure 1)...
[✓] 2026-04-09 14:30:01 提取完成: ./content/blog/posts/014_normformer_paper_review/figure_1.png
[INFO] 2026-04-09 14:30:01 Figure 1 验证结果:
  ✓ 宽度: 1275px
  ✓ 高度: 900px
  ✓ 文件大小: 184.0KB
  ✓ 压缩率: 0.1593 bytes/pixel
  质量评级: ⭐⭐⭐⭐⭐ 优秀 (极好压缩)
[✓] 2026-04-09 14:30:01 Figure 1 验证通过

...

[✓] 2026-04-09 14:30:03 质量报告已生成: ./content/blog/posts/014_normformer_paper_review/figures_quality_report.md
[INFO] 2026-04-09 14:30:03 ==========================================
[INFO] 2026-04-09 14:30:03 提取完成
[INFO] 2026-04-09 14:30:03 ==========================================
[✓] 2026-04-09 14:30:03 成功: 3 / 3
[✓] 2026-04-09 14:30:03 所有图表已成功提取和验证
```

### 一键提取 (Python 脚本)

```bash
# 从 NormFormer 论文提取第2、6、7页
python3 ./scripts/extract_paper_figures.py \
  --pdf /tmp/normformer.pdf \
  --pages 2,6,7 \
  --output-dir ./content/blog/posts/014_normformer_paper_review
```

---

## 工具对比（Bash vs Python）

| 维度 | Bash 脚本 | Python 脚本 |
|------|---------|-----------|
| **依赖** | 仅系统工具（轻量） | Python 3.6+ + Pillow |
| **安装** | 直接运行 | `pip install Pillow` |
| **性能** | 快速（直接调用系统命令） | 稍慢（Python 开销） |
| **错误处理** | 基础 | 强大（异常处理） |
| **可读性** | 中等 | 强（面向对象） |
| **扩展性** | 弱（脚本局限） | 强（可导入模块） |
| **跨平台** | 弱（macOS/Linux） | 强（Win/Mac/Linux） |
| **推荐场景** | 博客作者一键执行 | CI/CD 集成、批量处理 |

### 推荐搭配方案

**方案 A: 轻量级（单篇论文博客）**
- 使用 **Bash 脚本** 一键提取
- 检查 `figures_quality_report.md` 验证质量
- 修改 Markdown 中的图片引用

**方案 B: 自动化（多篇论文、CI/CD）**
- 使用 **Python 脚本** 作为基础模块
- 在 GitHub Actions 中集成
- 支持 CSV 批量处理
- 生成统一的质量报告

---

## 详细使用说明

### Bash 脚本使用

#### 基本用法

```bash
./scripts/extract_paper_figures.sh \
  --pdf <PDF_FILE> \
  --pages <PAGE_NUMBERS> \
  --output-dir <OUTPUT_DIR> \
  [OPTIONS]
```

#### 完整参数列表

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--pdf PATH` | ✓ | 输入 PDF 文件路径 | - |
| `--pages PAGES` | ✓ | 页码，逗号分隔 (如 "2,6,7") | - |
| `--output-dir DIR` | ✓ | 输出目录 | - |
| `--quality-threshold N` | | bytes/pixel 阈值 | 0.5 |
| `--min-width N` | | 最小宽度 (px) | 600 |
| `--min-height N` | | 最小高度 (px) | 400 |
| `--max-size N` | | 最大文件大小 (字节) | 500000 |
| `--no-report` | | 不生成质量报告 | 启用 |
| `--verbose` | | 输出调试信息 | 禁用 |
| `--help` | | 显示帮助 | - |

#### 实际案例

**案例 1: 标准论文提取（NormFormer）**

```bash
./scripts/extract_paper_figures.sh \
  --pdf /tmp/2110.09456.pdf \
  --pages "2,6,7" \
  --output-dir ./content/blog/posts/014_normformer_paper_review
```

**案例 2: 严格的质量要求**

```bash
./scripts/extract_paper_figures.sh \
  --pdf paper.pdf \
  --pages "1,3,5,7" \
  --output-dir ./figures \
  --quality-threshold 0.2 \
  --min-width 800 \
  --max-size 300000
```

**案例 3: 调试模式（遇到问题时）**

```bash
./scripts/extract_paper_figures.sh \
  --pdf paper.pdf \
  --pages "2,6" \
  --output-dir ./output \
  --verbose
```

### Python 脚本使用

#### 基本用法

```bash
python3 ./scripts/extract_paper_figures.py \
  --pdf <PDF_FILE> \
  --pages <PAGE_NUMBERS> \
  --output-dir <OUTPUT_DIR> \
  [OPTIONS]
```

#### 参数同 Bash 脚本

#### 作为模块导入（高级用法）

```python
from pathlib import Path
import sys

sys.path.insert(0, './scripts')
from extract_paper_figures import (
    PaperFigureExtractor,
    PDFProcessor,
    ImageValidator,
    ReportGenerator
)

# 自定义参数对象
class Args:
    pdf = '/tmp/paper.pdf'
    pages = '2,6,7'
    output_dir = './figures'
    quality_threshold = 0.3
    min_width = 600
    min_height = 400
    max_size = 500000
    no_report = False
    verbose = True

args = Args()
extractor = PaperFigureExtractor(args)
exit_code = extractor.run()
```

---

## 与 Hugo 工作流集成

### 步骤 1: 下载论文 PDF

```bash
# 从 arXiv 下载
curl -L https://arxiv.org/pdf/2110.09456.pdf -o /tmp/normformer.pdf

# 或从本地获取
cp ~/Downloads/paper.pdf /tmp/paper.pdf
```

### 步骤 2: 运行提取脚本

```bash
BLOG_ID="014_normformer_paper_review"
PDF_FILE="/tmp/normformer.pdf"
OUTPUT_DIR="./content/blog/posts/$BLOG_ID"

./scripts/extract_paper_figures.sh \
  --pdf "$PDF_FILE" \
  --pages "2,6,7" \
  --output-dir "$OUTPUT_DIR"
```

### 步骤 3: 验证质量报告

```bash
cat ./content/blog/posts/$BLOG_ID/figures_quality_report.md
```

生成的报告示例:

```markdown
# 论文图表提取质量报告

**生成时间**: 2026-04-09 14:30:00
**输入 PDF**: /tmp/normformer.pdf
**输出目录**: ./content/blog/posts/014_normformer_paper_review

## 质量标准

| 指标 | 标准 |
|------|------|
| 最小宽度 | 600px |
| 最小高度 | 400px |
| 最大文件大小 | 500KB |
| bytes/pixel 阈值 | 0.5 |

## 提取结果

### Figure 1: figure_1.png
- **尺寸**: 1275×900px
- **文件大小**: 184.0KB
- **bytes/pixel**: 0.1593
- **质量评级**: ⭐⭐⭐⭐⭐ 优秀 (极好压缩)
- **状态**: ✓ 通过

...
```

### 步骤 4: 更新 Markdown 文件

在博客的 `index.md` 中添加或更新图片引用:

```markdown
## 架构设计

![Figure 1: 架构对比](figure_1.png)

本论文提出了 NormFormer 架构，相比 Transformer 有以下改进...

## 实验结果

![Figure 3: 梯度分布](figure_3.png)

从图3可以看出...

![Figure 4 & 5: 缩放参数和学习率](figure_4_5.png)

缩放参数的设置对模型训练有重要影响...
```

### 步骤 5: 本地预览

```bash
hugo server
# 访问 http://localhost:1313
```

### 步骤 6: 提交并推送

```bash
# 检查文件
git status

# 添加文件
git add content/blog/posts/014_normformer_paper_review/

# 提交
git commit -m "docs: 为NormFormer博客添加论文图表（Figure 1,3,4,5）"

# 推送
git push
```

---

## 批量处理

### 使用 CSV 批量提取多篇论文

创建 `papers.csv`:

```csv
pdf_file,pages,blog_id,title
/tmp/2110.09456.pdf,"2,6,7",014_normformer_paper_review,NormFormer论文精读
/tmp/2106.09786.pdf,"3,4,5,6",015_new_paper_review,新论文精读
/tmp/2103.10385.pdf,"1,2,3",013_another_paper,另一篇论文
```

创建批量处理脚本 `batch_extract.sh`:

```bash
#!/bin/bash

CSV_FILE="papers.csv"
SCRIPT_DIR="./scripts"

# 跳过 CSV 头
tail -n +2 "$CSV_FILE" | while IFS=',' read -r pdf_file pages blog_id title; do
    # 去除引号和空格
    pdf_file=$(echo "$pdf_file" | tr -d '"' | xargs)
    pages=$(echo "$pages" | tr -d '"' | xargs)
    blog_id=$(echo "$blog_id" | tr -d '"' | xargs)
    title=$(echo "$title" | tr -d '"' | xargs)

    OUTPUT_DIR="./content/blog/posts/$blog_id"

    echo "=========================================="
    echo "处理: $title"
    echo "=========================================="

    "$SCRIPT_DIR/extract_paper_figures.sh" \
        --pdf "$pdf_file" \
        --pages "$pages" \
        --output-dir "$OUTPUT_DIR"

    if [ $? -eq 0 ]; then
        echo "✓ $title 提取成功"
    else
        echo "❌ $title 提取失败"
    fi

    echo ""
done
```

运行:

```bash
chmod +x batch_extract.sh
./batch_extract.sh
```

### 使用 Python 批量处理

创建 `batch_extract.py`:

```python
#!/usr/bin/env python3

import csv
import sys
from pathlib import Path
from extract_paper_figures import PaperFigureExtractor

class Args:
    def __init__(self, pdf, pages, output_dir):
        self.pdf = pdf
        self.pages = pages
        self.output_dir = output_dir
        self.quality_threshold = 0.5
        self.min_width = 600
        self.min_height = 400
        self.max_size = 500000
        self.no_report = False
        self.verbose = False

def main():
    csv_file = 'papers.csv'

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdf_file = row['pdf_file'].strip()
            pages = row['pages'].strip()
            blog_id = row['blog_id'].strip()
            title = row['title'].strip()

            output_dir = f"./content/blog/posts/{blog_id}"

            print(f"\n{'=' * 50}")
            print(f"处理: {title}")
            print(f"{'=' * 50}\n")

            args = Args(pdf_file, pages, output_dir)
            extractor = PaperFigureExtractor(args)

            if extractor.run() == 0:
                print(f"✓ {title} 提取成功")
            else:
                print(f"❌ {title} 提取失败")

if __name__ == '__main__':
    main()
```

运行:

```bash
chmod +x batch_extract.py
python3 batch_extract.py
```

---

## 常见问题 FAQ

### Q1: 我如何确定 PDF 中哪些页面包含我要的 Figure?

**A**: 使用 PDF 查看器（Preview/Acrobat）查看 PDF，记下页码。也可以用命令预览:

```bash
# 生成第 2 页的预览 (需要 pdftoppm)
pdftoppm -f 2 -l 2 -png -r 100 paper.pdf page_preview

# 用 Preview 查看
open page_preview-002.png
```

### Q2: 脚本说 pdfimages 失败，自动降级到 pdftoppm，这是正常的吗?

**A**: 完全正常。这说明:
- PDF 中没有嵌入图片（可能是矢量图或纯文本）
- pdfimages 不支持的 PDF 格式

降级方案会使用 `pdftoppm` 渲染整个页面，文件可能会大一些（通常 300-500KB），但质量仍可接受。如果需要优化，可以:

1. 手工在图像编辑器中裁剪去除余白
2. 调整 pdftoppm 的分辨率参数（见故障排查）

### Q3: 质量评级是什么意思？我应该对所有 Figure 都要求 ⭐⭐⭐⭐⭐?

**A**: 不必。评级基于 `bytes/pixel` 比率，表示压缩效率:

- **⭐⭐⭐⭐⭐**: < 0.15 bytes/pixel - 非常好的压缩
- **⭐⭐⭐⭐**: 0.15-0.25 - 很好
- **⭐⭐⭐**: 0.25-0.5 - 可接受（建议标准）
- **⭐⭐**: 0.5-1.0 - 一般，需要优化
- **⭐**: > 1.0 - 太大，必须优化

对于博客来说，**⭐⭐⭐ 及以上就满足要求**。NormFormer 论文的 Figure 都达到了 ⭐⭐⭐⭐⭐ 的标准。

### Q4: 文件大小限制 500KB 是固定的吗？我可以调整吗?

**A**: 可以。使用 `--max-size` 参数调整:

```bash
# 允许更大的文件 (1MB)
./scripts/extract_paper_figures.sh \
  --pdf paper.pdf \
  --pages "2,6" \
  --output-dir ./output \
  --max-size 1000000

# 更严格的限制 (300KB)
./scripts/extract_paper_figures.sh \
  --pdf paper.pdf \
  --pages "2,6" \
  --output-dir ./output \
  --max-size 300000
```

建议:
- **博客展示**: 100-300KB (平衡速度和质量)
- **学术文档**: 500KB-1MB (允许更高质量)
- **对带宽敏感**: 50-150KB (严格压缩)

### Q5: 提取出的图片有些模糊，能改进吗?

**A**: 这通常意味着 PDF 本身的嵌图质量不高，或者使用了 pdftoppm 降级渲染。尝试:

1. **调整 pdftoppm 分辨率** (默认 150 DPI)
   ```bash
   # 在脚本中找到这一行，改成更高分辨率
   # pdftoppm -f $page -l $page -png -r 150 ...
   # 改成 -r 200 或 -r 300
   ```

2. **在图像编辑器中手工优化**
   ```bash
   # 用 ImageMagick 增强清晰度
   convert figure_1.png -sharpen 0x1 figure_1_sharp.png
   ```

3. **检查源 PDF 的质量**
   ```bash
   # 用 pdfimages 列出嵌图信息
   pdfimages -list paper.pdf
   # 查看 "image" 列，通常会显示 colorspace 和压缩方式
   ```

### Q6: 我可以一次提取 PDF 的所有页面吗?

**A**: 可以，但不建议这样做（会生成很多图片）。如果真的需要:

```bash
# 获取 PDF 页数
TOTAL_PAGES=$(pdfinfo paper.pdf | grep Pages | awk '{print $2}')

# 生成页码列表
PAGES=$(seq 1 $TOTAL_PAGES | paste -sd ',' -)

# 提取所有页
./scripts/extract_paper_figures.sh \
  --pdf paper.pdf \
  --pages "$PAGES" \
  --output-dir ./output
```

但通常你只需要包含关键 Figure 的页面（通常是 2-5 个）。

### Q7: 支持 JPG 输出吗? 我想减小文件大小。

**A**: 脚本目前输出 PNG (更好的压缩)。如果需要 JPG:

```bash
# 手工转换
convert figure_1.png -quality 85 figure_1.jpg

# 或在脚本中修改 (高级用法)
# 在脚本最后添加转换步骤
```

**不建议用 JPG**，因为:
- PNG 对于图表、截图更有效
- JPG 会引入块状伪影
- PNG 的压缩比例其实不差（见 NormFormer 的 bytes/pixel 数据）

### Q8: 在 Windows 上能运行吗?

**A**: 

- **Bash 脚本**: 不行，需要 WSL2 或 MINGW
- **Python 脚本**: 可以，需要安装依赖:

```bash
# Windows cmd
pip install Pillow

# 确保安装了 poppler
# 从 https://github.com/oschwartz10612/poppler-windows/releases/ 下载
# 或用 Chocolatey: choco install poppler
```

然后运行:

```bash
python scripts/extract_paper_figures.py --pdf paper.pdf --pages 2,6 --output-dir output
```

### Q9: CI/CD 集成（GitHub Actions）怎么做?

**A**: 创建 `.github/workflows/extract_figures.yml`:

```yaml
name: Extract Paper Figures

on:
  push:
    paths:
      - 'papers/**'
  workflow_dispatch:

jobs:
  extract:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y imagemagick poppler-utils
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Extract figures
        run: |
          python3 scripts/extract_paper_figures.py \
            --pdf papers/latest.pdf \
            --pages "2,6,7" \
            --output-dir "content/blog/posts/latest"
      
      - name: Commit and push
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add -A
          git diff --cached --exit-code || git commit -m "auto: extract paper figures"
          git push
```

### Q10: 我想要自定义输出的文件名格式（如 Fig_1_hq.png）

**A**: 编辑脚本，找到这一行:

```bash
# Bash 脚本，大约第 360 行
local target_file="$OUTPUT_DIR/figure_${figure_num}.png"

# 改成
local target_file="$OUTPUT_DIR/fig_${figure_num}_hq.png"
```

或者提取后手工重命名:

```bash
cd content/blog/posts/014_normformer_paper_review/
mv figure_1.png fig_1_hq.png
mv figure_3.png fig_3_hq.png
mv figure_4_5.png fig_45_hq.png
```

---

## 故障排查

### 问题 1: "Command not found: pdfimages"

**症状**:
```
./extract_paper_figures.sh: line 123: pdfimages: command not found
```

**原因**: 未安装 poppler 工具集

**解决**:
```bash
# macOS
brew install poppler

# Ubuntu
sudo apt-get install poppler-utils

# CentOS
sudo yum install poppler-utils

# 验证安装
pdfimages --version
```

### 问题 2: "convert: command not found"

**症状**:
```
convert: command not found
```

**原因**: 未安装 ImageMagick

**解决**:
```bash
# macOS
brew install imagemagick

# Ubuntu
sudo apt-get install imagemagick

# 验证安装
convert --version
```

### 问题 3: "PPM 转换失败"

**症状**: pdfimages 成功但转换 PPM 到 PNG 失败

**原因**: 文件损坏或格式问题

**解决**:
```bash
# 检查 PPM 文件
file temp_dir/page_2_*

# 手工转换调试
convert -verbose temp_dir/page_2_000.ppm temp_dir/page_2_000.png

# 如果还是失败，尝试降级参数
convert -quality 85 temp_dir/page_2_000.ppm temp_dir/page_2_000.png
```

### 问题 4: "输出文件为空"

**症状**: 脚本运行成功但输出目录没有文件

**原因**: 通常是 PDF 页码错误或权限问题

**解决**:
```bash
# 验证 PDF 页数
pdfinfo paper.pdf | grep Pages

# 验证输出目录权限
ls -la content/blog/posts/

# 使用详细模式重试
./scripts/extract_paper_figures.sh \
  --pdf paper.pdf \
  --pages "2,6" \
  --output-dir ./output \
  --verbose
```

### 问题 5: "文件大小过大（> 500KB）"

**症状**: 提取的图片超过限制

**原因**: 
- PDF 中的图片质量非常高
- pdftoppm 降级方案使用了低分辨率

**解决**:
```bash
# 方案 1: 调整大小限制
./scripts/extract_paper_figures.sh \
  --pdf paper.pdf \
  --pages "2,6" \
  --output-dir ./output \
  --max-size 800000

# 方案 2: 压缩图片
convert figure_1.png -quality 80 -strip figure_1_compressed.png

# 方案 3: 裁剪去除余白
convert figure_1.png -trim +repage figure_1_trimmed.png
```

### 问题 6: "MacOS 上 pdfimages 输出是 PPM 而非 PNG"

**症状**: pdfimages 生成 `.ppm` 文件而不是 `.png`

**原因**: macOS 的 poppler 版本行为不同

**解决**: 这是正常的，脚本会自动转换。如果转换失败:

```bash
# 手工转换
convert page_2_-000.ppm page_2_000.png

# 或调试脚本
./scripts/extract_paper_figures.sh \
  --pdf paper.pdf \
  --pages "2" \
  --output-dir ./output \
  --verbose
```

### 问题 7: "权限被拒绝 (Permission denied)"

**症状**:
```
./extract_paper_figures.sh: Permission denied
```

**原因**: 脚本没有执行权限

**解决**:
```bash
# 添加执行权限
chmod +x scripts/extract_paper_figures.sh

# 或直接用 bash 运行
bash scripts/extract_paper_figures.sh --pdf paper.pdf --pages 2,6 --output-dir output
```

### 问题 8: Python 脚本导入错误

**症状**:
```
ModuleNotFoundError: No module named 'PIL'
```

**原因**: 未安装 Pillow

**解决**:
```bash
pip install Pillow

# 或用系统包管理
# Ubuntu: sudo apt-get install python3-pil
# macOS: brew install python-pillow
```

### 问题 9: 脚本在大型 PDF 上很慢

**症状**: 处理超过 100MB 的 PDF 时很慢

**原因**: 系统资源限制或 PDF 结构复杂

**解决**:
```bash
# 只提取需要的页面，避免处理整个 PDF
./scripts/extract_paper_figures.sh \
  --pdf large_paper.pdf \
  --pages "2,6,7" \
  --output-dir ./output
# 不要尝试 --pages "1,2,3,...,100"

# 如果只需要某几个 Figure，考虑先分割 PDF
# 用 pdfseparate 或 Python pdfplumber 库
```

### 问题 10: 生成的图片在 Hugo 中显示不正确

**症状**: Hugo 服务器预览时图片显示错误或变形

**原因**: 
- 图片引用路径错误
- Hugo 缓存问题

**解决**:
```bash
# 清理 Hugo 缓存
rm -rf resources/_gen

# 检查引用路径 (应该是相对路径)
# 正确: ![Figure](figure_1.png)
# 错误: ![Figure](./figure_1.png) 或 ![Figure](/figure_1.png)

# 重新启动 Hugo 服务
hugo server --disableFastRender
```

---

## 总结

### 快速参考卡

```bash
# ============ 基本用法 ============

# 提取单篇论文的多个 Figure
./scripts/extract_paper_figures.sh \
  --pdf /tmp/paper.pdf \
  --pages "2,6,7" \
  --output-dir ./content/blog/posts/blog_id

# 严格质量要求
./scripts/extract_paper_figures.sh \
  --pdf paper.pdf \
  --pages "2,6" \
  --output-dir ./output \
  --quality-threshold 0.3 \
  --max-size 300000

# 调试模式
./scripts/extract_paper_figures.sh \
  --pdf paper.pdf \
  --pages "2" \
  --output-dir ./output \
  --verbose

# ============ 集成到工作流 ============

# 1. 下载 PDF
curl -L https://arxiv.org/pdf/XXXX.XXXXX.pdf -o /tmp/paper.pdf

# 2. 运行脚本
./scripts/extract_paper_figures.sh \
  --pdf /tmp/paper.pdf \
  --pages "..." \
  --output-dir ./content/blog/posts/blog_id

# 3. 查看报告
cat ./content/blog/posts/blog_id/figures_quality_report.md

# 4. 编辑 Markdown 文件并提交
git add content/blog/posts/blog_id/
git commit -m "docs: 添加论文 Figure"
git push

# ============ 批量处理 ============

# 使用 CSV 文件批量提取
./batch_extract.sh  # 或 python3 batch_extract.py

# ============ 质量检查 ============

# 查看提取结果的质量报告
cat content/blog/posts/blog_id/figures_quality_report.md

# 如果质量不满足，手工优化
convert figure_1.png -trim +repage figure_1_trimmed.png
```

