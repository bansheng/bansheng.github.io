#!/bin/bash
# Batch generate all blog covers using v3 photography-based template

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GENERATOR="$SCRIPT_DIR/generate-cover-v3.js"

# Article configuration: [dir] [title] [tags] [photo-seed]
# Format: directory name | article title | tags (comma-separated) | unique photo seed
declare -a articles=(
  "001_my_first_post|我的第一篇博客|科技,AI,技术博客|101"
  "002_scaling_laws_2026|Scaling Laws 2026：为什么更大更好|神经网络,Scaling Laws,深度学习|202"
  "003_how_i_built_my_blog|我如何用 HugoBlox 构建学术博客|Hugo,静态网站,博客|303"
  "004_openclaw_zhipu_api_setup|OpenClaw + 智谱 API 本地开发环境搭建|API,工具配置,大模型|404"
  "005_tokenmixer_large_paper_review|TokenMixer：用隐式令牌混合器扩展 Transformer|论文精读,Transformer,架构设计|505"
  "006_fuxi_linear_paper_review|FUXI Linear：通向线性 Transformer 的新路径|论文精读,Transformer,线性注意力|606"
  "007_hstu_paper_review|HSTU：分层序列转导单元|论文精读,推荐系统,Transformer|707"
  "008_mfalcon_hstu_inference|mFalcon：从 HSTU 到工业级推理|推理优化,系统优化,推荐系统|808"
  "009_gpsd_paper_review|GPSD：生成式预训练推荐系统|论文精读,生成模型,推荐系统|909"
  "010_mtfm_meituan_foundation_model|MTFM：美团基础模型架构|论文精读,基础模型,推荐系统|1010"
  "011_sort_paper_review|SORT：面向工业级推荐系统的系统优化排序 Transformer|论文精读,推荐系统,Transformer|1111"
)

echo "🎨 Batch generating blog covers (v3 - photography + text)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

COUNT=0
for article in "${articles[@]}"; do
  IFS='|' read -r dir title tags seed <<< "$article"

  output_path="$PROJECT_ROOT/content/blog/posts/$dir/featured.png"

  echo "[$((++COUNT))/11] Generating: $title"
  echo "  → $output_path"

  node "$GENERATOR" \
    --title "$title" \
    --tags "$tags" \
    --photo-seed "$seed" \
    --output "$output_path"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All 11 covers generated successfully!"
echo ""
echo "Next: Run 'hugo server' to preview locally"
