#!/usr/bin/env node
/**
 * Blog Cover Generator V3 - Photography + Text Overlay
 *
 * Usage:
 *   node scripts/generate-cover-v3.js \
 *     --title "SORT：工业级推荐排序 Transformer" \
 *     --tags "论文精读,推荐系统,MoE" \
 *     --photo-seed 1111 \
 *     --output content/blog/posts/011_sort_paper_review/featured.png
 *
 * Features:
 *   - Real photography background from picsum.photos
 *   - Gradient overlay for text readability
 *   - Bottom-left text layout with title and tags
 */

const puppeteer = require('puppeteer');
const path = require('path');

const args = process.argv.slice(2);
function getArg(name) {
  const idx = args.indexOf(`--${name}`);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : null;
}

const title = getArg('title') || 'Untitled';
const tags = (getArg('tags') || '').split(',').filter(Boolean);
const photoSeed = getArg('photo-seed') || '42';
const output = getArg('output') || 'featured.png';

const tagsHtml = tags.map(tag =>
  `<span style="display:inline-block;padding:5px 12px;background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.3);border-radius:16px;font-size:12px;font-weight:600;color:white;backdrop-filter:blur(4px);">${tag}</span>`
).join('');

const html = `<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;width:1200px;height:630px;overflow:hidden;background:#f0f0f0;">
<div style="
  width:1200px;height:630px;
  position:relative;
  overflow:hidden;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'PingFang SC','Microsoft YaHei',sans-serif;
">
  <!-- Photography background -->
  <img src="https://picsum.photos/seed/${photoSeed}/1200/630?random=1"
       style="position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;object-position:center;" />

  <!-- Gradient overlay (bottom to top: dark to transparent) -->
  <div style="
    position:absolute;
    top:0;left:0;right:0;bottom:0;
    background:linear-gradient(to top, rgba(0,0,0,0.75) 0%, rgba(0,0,0,0.5) 30%, rgba(0,0,0,0.2) 60%, rgba(0,0,0,0) 100%);
    z-index:10;
  "></div>

  <!-- Content: bottom-left layout -->
  <div style="
    position:absolute;
    bottom:40px;left:60px;right:60px;
    z-index:20;
    color:white;
  ">
    <!-- Article title (large, bold) -->
    <h1 style="
      margin:0 0 12px 0;
      font-size:36px;
      font-weight:900;
      line-height:1.15;
      text-shadow:0 2px 8px rgba(0,0,0,0.4);
      color:white;
    ">${title}</h1>

    <!-- Tags -->
    <div style="
      display:flex;
      gap:6px;
      flex-wrap:wrap;
    ">${tagsHtml}</div>
  </div>
</div>
</body>
</html>`;

(async () => {
  try {
    const browser = await puppeteer.launch({
      headless: 'new',
      args: ['--no-sandbox']
    });
    const page = await browser.newPage();
    await page.setViewport({ width: 1200, height: 630, deviceScaleFactor: 2 });
    await page.setContent(html, { waitUntil: 'networkidle0' });
    await page.screenshot({ path: output, type: 'png' });
    await browser.close();
    console.log(`✓ Cover generated: ${output}`);
    console.log(`  Title: ${title}`);
    console.log(`  Photo seed: ${photoSeed}`);
  } catch (error) {
    console.error('Error generating cover:', error.message);
    process.exit(1);
  }
})();
