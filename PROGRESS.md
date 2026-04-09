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
