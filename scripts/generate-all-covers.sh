#!/bin/bash
# Light theme covers with architecture diagrams
cd "$(dirname "$0")/.."
GEN="node scripts/generate-cover-v2.js"

# Light color palette for boxes
# blue=#dbeafe  purple=#ede9fe  green=#d1fae5  orange=#fef3c7  red=#fee2e2  cyan=#cffafe
# text on boxes: blue=#1e40af  purple=#6d28d9  green=#047857  orange=#b45309  red=#b91c1c  cyan=#0e7490

$GEN --title "Hello World" --subtitle "OpenClaw + Hugo 自动生成的第一篇博客" \
  --diagram '<div style="display:flex;align-items:center;gap:14px;">
    <div style="padding:12px 20px;background:#e0f2fe;border:1px solid #bae6fd;border-radius:12px;font-size:14px;font-weight:700;color:#0369a1;">🧑‍💻 Dev</div>
    <div style="color:#94a3b8;font-size:20px;">→</div>
    <div style="padding:12px 20px;background:#fef3c7;border:1px solid #fde68a;border-radius:12px;font-size:14px;font-weight:700;color:#b45309;">🤖 OpenClaw</div>
    <div style="color:#94a3b8;font-size:20px;">→</div>
    <div style="padding:12px 20px;background:#d1fae5;border:1px solid #a7f3d0;border-radius:12px;font-size:14px;font-weight:700;color:#047857;">⚡ Hugo</div>
    <div style="color:#94a3b8;font-size:20px;">→</div>
    <div style="padding:12px 20px;background:#ede9fe;border:1px solid #ddd6fe;border-radius:12px;font-size:14px;font-weight:700;color:#6d28d9;">🌐 Blog</div>
  </div>' \
  --output content/blog/posts/001_my_first_post/featured.png --theme cyan

$GEN --title "Scaling Laws" --subtitle "2026年：从规模竞赛到效率革命" \
  --diagram '<div style="display:flex;flex-direction:column;gap:10px;align-items:center;">
    <div style="padding:12px 24px;background:#f3f4f6;border:1px solid #e5e7eb;border-radius:12px;font-size:14px;font-weight:600;color:#374151;">📈 More Params + Data</div>
    <div style="color:#d1d5db;font-size:12px;">diminishing returns ↓</div>
    <div style="padding:10px 20px;background:#fef3c7;border:1px solid #fde68a;border-radius:20px;font-size:13px;font-weight:700;color:#b45309;">🔄 Paradigm Shift</div>
    <div style="display:flex;gap:8px;margin-top:4px;">
      <div style="padding:10px 14px;background:#dbeafe;border:1px solid #bfdbfe;border-radius:10px;font-size:11px;font-weight:700;color:#1e40af;">🧠 Test-time</div>
      <div style="padding:10px 14px;background:#d1fae5;border:1px solid #a7f3d0;border-radius:10px;font-size:11px;font-weight:700;color:#047857;">💎 Quality</div>
      <div style="padding:10px 14px;background:#ede9fe;border:1px solid #ddd6fe;border-radius:10px;font-size:11px;font-weight:700;color:#6d28d9;">🌐 Multimodal</div>
    </div>
  </div>' \
  --output content/blog/posts/002_scaling_laws_2026/featured.png --theme purple

$GEN --title "博客搭建" --subtitle "Hugo + HugoBlox + Giscus + GitHub Pages" \
  --diagram '<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;justify-content:center;">
    <div style="padding:10px 14px;background:#cffafe;border:1px solid #a5f3fc;border-radius:10px;font-size:12px;font-weight:700;color:#0e7490;">🔧 Hugo</div>
    <div style="color:#d1d5db;">→</div>
    <div style="padding:10px 14px;background:#ede9fe;border:1px solid #ddd6fe;border-radius:10px;font-size:12px;font-weight:700;color:#6d28d9;">🎨 Theme</div>
    <div style="color:#d1d5db;">→</div>
    <div style="padding:10px 14px;background:#d1fae5;border:1px solid #a7f3d0;border-radius:10px;font-size:12px;font-weight:700;color:#047857;">💬 Giscus</div>
    <div style="color:#d1d5db;">→</div>
    <div style="padding:10px 14px;background:#fef3c7;border:1px solid #fde68a;border-radius:10px;font-size:12px;font-weight:700;color:#b45309;">⚙️ CI/CD</div>
    <div style="color:#d1d5db;">→</div>
    <div style="padding:10px 14px;background:#dbeafe;border:1px solid #bfdbfe;border-radius:10px;font-size:12px;font-weight:700;color:#1e40af;">🚀 Deploy</div>
  </div>' \
  --output content/blog/posts/003_how_i_built_my_blog/featured.png --theme cyan

$GEN --title "OpenClaw × 智谱" --subtitle "绕过 Coding Plan，免费接入智谱 AI API" \
  --diagram '<div style="display:flex;flex-direction:column;gap:10px;align-items:center;">
    <div style="padding:12px 22px;background:#d1fae5;border:1px solid #a7f3d0;border-radius:12px;font-size:14px;font-weight:700;color:#047857;">📝 注册智谱 AI</div>
    <div style="color:#d1d5db;">↓</div>
    <div style="padding:12px 22px;background:#fef3c7;border:1px solid #fde68a;border-radius:12px;font-size:14px;font-weight:700;color:#b45309;">🔑 获取 API Key</div>
    <div style="color:#d1d5db;">↓</div>
    <div style="padding:12px 22px;background:#dbeafe;border:1px solid #bfdbfe;border-radius:12px;font-size:14px;font-weight:700;color:#1e40af;">⚙️ 配置 OpenClaw</div>
    <div style="color:#d1d5db;">↓</div>
    <div style="padding:12px 22px;background:#ede9fe;border:1px solid #ddd6fe;border-radius:12px;font-size:14px;font-weight:700;color:#6d28d9;">🎉 免费 AI 编程</div>
  </div>' \
  --output content/blog/posts/004_openclaw_zhipu_api_setup/featured.png --theme green

$GEN --title "TokenMixer" --subtitle "工业级推荐系统的大模型扩展瓶颈突破" \
  --diagram '<div style="display:flex;gap:0;width:100%;height:100%;align-items:center;justify-content:center;">
    <div style="display:flex;flex-direction:column;gap:3px;align-items:center;">
      <div style="padding:10px 30px;background:#f3f4f6;border:2px solid #d1d5db;border-radius:12px;font-size:16px;font-weight:700;color:#374151;">Raw Sparse Features</div>
      <div style="color:#b0b0b0;font-size:15px;line-height:1;">↓</div>
      <div style="padding:9px 24px;background:#fff7ed;border:2px solid #fed7aa;border-radius:10px;font-size:15px;font-weight:700;color:#c2410c;">Tokenizer + Embedding</div>
      <div style="color:#b0b0b0;font-size:15px;line-height:1;">↓</div>
      <div style="display:flex;gap:5px;align-items:center;">
        <div style="padding:8px 14px;background:#fef3c7;border:2px solid #fde68a;border-radius:10px;font-size:14px;font-weight:700;color:#b45309;">Mixing T→H</div>
        <div style="font-size:15px;color:#b0b0b0;">⇄</div>
        <div style="padding:8px 14px;background:#ede9fe;border:2px solid #ddd6fe;border-radius:10px;font-size:14px;font-weight:700;color:#6d28d9;">Revert H→T</div>
      </div>
      <div style="color:#b0b0b0;font-size:15px;line-height:1;">↓</div>
      <div style="padding:7px 20px;background:#fef9c3;border:2px solid #fde047;border-radius:10px;font-size:14px;font-weight:700;color:#a16207;">Pre-RMSNorm</div>
      <div style="color:#b0b0b0;font-size:15px;line-height:1;">↓</div>
      <div style="padding:10px 24px;background:#dbeafe;border:2px solid #93c5fd;border-radius:12px;font-size:15px;font-weight:800;color:#1e40af;">Deep Stack (pSwiGLU)</div>
      <div style="color:#b0b0b0;font-size:15px;line-height:1;">↓</div>
      <div style="padding:9px 22px;background:#d1fae5;border:2px solid #6ee7b7;border-radius:10px;font-size:15px;font-weight:700;color:#047857;">Sparse Per-token MoE</div>
      <div style="color:#b0b0b0;font-size:15px;line-height:1;">↓</div>
      <div style="padding:8px 22px;background:#e0e7ff;border:2px solid #a5b4fc;border-radius:10px;font-size:14px;font-weight:700;color:#4338ca;">Mean Pooling</div>
      <div style="color:#b0b0b0;font-size:15px;line-height:1;">↓</div>
      <div style="padding:10px 30px;background:#f0fdf4;border:2px solid #86efac;border-radius:12px;font-size:16px;font-weight:800;color:#15803d;">Prediction</div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;width:32px;margin-left:6px;">
      <div style="font-size:9px;color:#d97706;font-weight:700;writing-mode:vertical-lr;letter-spacing:2px;margin-bottom:2px;">RESIDUAL</div>
      <div style="border-left:2px dashed #f59e0b;height:440px;position:relative;">
        <div style="position:absolute;top:-6px;left:-5px;color:#f59e0b;font-size:12px;">▲</div>
        <div style="position:absolute;bottom:-6px;left:-5px;color:#f59e0b;font-size:12px;">▼</div>
      </div>
    </div>
  </div>' \
  --output content/blog/posts/005_tokenmixer_large_paper_review/featured.png --theme orange

$GEN --title "FuXi-Linear" --subtitle "线性注意力释放超长序列推荐潜力" \
  --diagram '<div style="display:flex;flex-direction:column;gap:10px;align-items:center;">
    <div style="padding:10px 18px;background:#f3f4f6;border:1px solid #e5e7eb;border-radius:10px;font-size:13px;color:#374151;">User Sequence</div>
    <div style="display:flex;gap:8px;">
      <div style="padding:10px 14px;background:#dbeafe;border:1px solid #bfdbfe;border-radius:10px;font-size:11px;font-weight:700;color:#1e40af;">🧠 Semantic</div>
      <div style="padding:10px 14px;background:#fef3c7;border:1px solid #fde68a;border-radius:10px;font-size:11px;font-weight:700;color:#b45309;">⏱️ Temporal</div>
      <div style="padding:10px 14px;background:#ede9fe;border:1px solid #ddd6fe;border-radius:10px;font-size:11px;font-weight:700;color:#6d28d9;">📍 Positional</div>
    </div>
    <div style="color:#d1d5db;">↓ concat + gate</div>
    <div style="padding:10px 18px;background:#d1fae5;border:1px solid #a7f3d0;border-radius:10px;font-size:13px;font-weight:700;color:#047857;">🎯 Next Item Prediction</div>
  </div>' \
  --output content/blog/posts/006_fuxi_linear_paper_review/featured.png --theme blue

$GEN --title "HSTU" --subtitle "万亿参数推荐大模型：超越传统 Self-Attention" \
  --diagram '<div style="display:flex;flex-direction:column;gap:8px;align-items:center;">
    <div style="padding:10px 18px;background:#f3f4f6;border:1px solid #e5e7eb;border-radius:10px;font-size:13px;color:#374151;">Input Sequence N×d</div>
    <div style="color:#d1d5db;font-size:12px;">↓ LayerNorm → SiLU → Split</div>
    <div style="display:flex;gap:10px;">
      <div style="padding:12px 18px;background:#dbeafe;border:1px solid #bfdbfe;border-radius:10px;font-size:12px;font-weight:700;color:#1e40af;">🌐 Spatial<br/>Aggregation</div>
      <div style="padding:12px 18px;background:#fef3c7;border:1px solid #fde68a;border-radius:10px;font-size:12px;font-weight:700;color:#b45309;">🎯 Pointwise<br/>Transform</div>
    </div>
    <div style="color:#d1d5db;">↓ residual</div>
    <div style="padding:10px 18px;background:#d1fae5;border:1px solid #a7f3d0;border-radius:10px;font-size:13px;font-weight:700;color:#047857;">Output N×d</div>
  </div>' \
  --output content/blog/posts/007_hstu_paper_review/featured.png --theme purple

$GEN --title "M-FALCON" --subtitle "HSTU 推理加速：KV-Cache + 微批次推理" \
  --diagram '<div style="display:flex;flex-direction:column;gap:8px;align-items:center;">
    <div style="padding:10px 18px;background:#f3f4f6;border:1px solid #e5e7eb;border-radius:10px;font-size:13px;color:#374151;">👤 User History (n)</div>
    <div style="color:#d1d5db;">↓ pre-compute once</div>
    <div style="padding:12px 22px;background:#dbeafe;border:1px solid #bfdbfe;border-radius:12px;font-size:14px;font-weight:700;color:#1e40af;">📦 KV-Cache</div>
    <div style="color:#d1d5db;">↓ broadcast</div>
    <div style="padding:12px 22px;background:#fef3c7;border:1px solid #fde68a;border-radius:12px;font-size:14px;font-weight:700;color:#b45309;">🎭 Attention Mask</div>
    <div style="color:#d1d5db;">↓</div>
    <div style="padding:10px 18px;background:#d1fae5;border:1px solid #a7f3d0;border-radius:12px;font-size:13px;font-weight:700;color:#047857;">🏆 Parallel Ranking</div>
  </div>' \
  --output content/blog/posts/008_mfalcon_hstu_inference/featured.png --theme red

$GEN --title "GPSD" --subtitle "生成式预训练让判别式推荐也有 Scaling Law" \
  --diagram '<div style="display:flex;flex-direction:column;gap:10px;align-items:center;">
    <div style="padding:12px 22px;background:#dbeafe;border:1px solid #bfdbfe;border-radius:12px;font-size:14px;font-weight:700;color:#1e40af;">Phase 1: 生成式预训练</div>
    <div style="color:#d1d5db;">↓ transfer</div>
    <div style="display:flex;gap:10px;">
      <div style="padding:10px 14px;background:#fef3c7;border:1px solid #fde68a;border-radius:10px;font-size:12px;font-weight:700;color:#b45309;">🔒 Freeze<br/>Embedding</div>
      <div style="padding:10px 14px;background:#ede9fe;border:1px solid #ddd6fe;border-radius:10px;font-size:12px;font-weight:700;color:#6d28d9;">🚀 Migrate<br/>Transformer</div>
    </div>
    <div style="color:#d1d5db;">↓</div>
    <div style="padding:12px 22px;background:#d1fae5;border:1px solid #a7f3d0;border-radius:12px;font-size:14px;font-weight:700;color:#047857;">Phase 2: 判别式训练</div>
  </div>' \
  --output content/blog/posts/009_gpsd_paper_review/featured.png --theme green

$GEN --title "MTFM" --subtitle "美团多场景推荐基座模型深度解析" \
  --diagram '<div style="display:flex;flex-direction:column;gap:10px;align-items:center;">
    <div style="display:flex;gap:8px;">
      <div style="padding:10px 14px;background:#f3f4f6;border:1px solid #e5e7eb;border-radius:10px;font-size:12px;font-weight:600;color:#374151;">🍔 外卖</div>
      <div style="padding:10px 14px;background:#f3f4f6;border:1px solid #e5e7eb;border-radius:10px;font-size:12px;font-weight:600;color:#374151;">🚲 骑行</div>
      <div style="padding:10px 14px;background:#f3f4f6;border:1px solid #e5e7eb;border-radius:10px;font-size:12px;font-weight:600;color:#374151;">🛍️ 购物</div>
    </div>
    <div style="color:#d1d5db;font-size:12px;">↓ heterogeneous tokenization</div>
    <div style="padding:12px 24px;background:#fef3c7;border:1px solid #fde68a;border-radius:12px;font-size:14px;font-weight:700;color:#b45309;">✨ Unified Sequence</div>
    <div style="color:#d1d5db;">↓</div>
    <div style="padding:12px 24px;background:#ede9fe;border:1px solid #ddd6fe;border-radius:12px;font-size:14px;font-weight:700;color:#6d28d9;">🔥 Single Transformer</div>
  </div>' \
  --output content/blog/posts/010_mtfm_meituan_foundation_model/featured.png --theme orange

$GEN --title "SORT" --subtitle "面向工业级推荐系统的系统优化排序 Transformer" \
  --diagram '<div style="display:flex;flex-direction:column;gap:8px;align-items:center;">
    <div style="display:flex;gap:12px;">
      <div style="padding:10px 18px;background:#f3f4f6;border:1px solid #e5e7eb;border-radius:10px;font-size:13px;font-weight:600;color:#374151;">📜 History</div>
      <div style="padding:10px 18px;background:#f3f4f6;border:1px solid #e5e7eb;border-radius:10px;font-size:13px;font-weight:600;color:#374151;">📋 Candidates</div>
    </div>
    <div style="color:#d1d5db;">↓</div>
    <div style="padding:12px 24px;background:#dbeafe;border:1px solid #bfdbfe;border-radius:10px;font-size:13px;font-weight:700;color:#1e40af;">Layer 1-4 · Full Attention</div>
    <div style="color:#d1d5db;">↓</div>
    <div style="padding:12px 24px;background:#fef3c7;border:1px solid #fde68a;border-radius:10px;font-size:13px;font-weight:700;color:#b45309;">Layer 5-8 · ✂️ Prune Q</div>
    <div style="color:#d1d5db;">↓</div>
    <div style="padding:12px 24px;background:#ede9fe;border:1px solid #ddd6fe;border-radius:10px;font-size:13px;font-weight:700;color:#6d28d9;">Layer 9-12 · MoE FFN</div>
    <div style="color:#d1d5db;">↓</div>
    <div style="padding:10px 20px;background:#d1fae5;border:1px solid #a7f3d0;border-radius:10px;font-size:13px;font-weight:700;color:#047857;">🎯 CTR +0.26pt</div>
  </div>' \
  --output content/blog/posts/011_sort_paper_review/featured.png --theme blue

echo "All 11 light-theme covers generated!"
