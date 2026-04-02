#!/usr/bin/env node
/**
 * Blog Cover Generator
 *
 * Usage:
 *   node scripts/generate-cover.js --title "SORT" --subtitle "工业级排序 Transformer" --tags "推荐系统,Transformer,MoE" --output content/blog/posts/011_sort_paper_review/featured.png --theme blue
 *
 * Themes: blue, purple, green, orange, red, cyan
 */

const puppeteer = require('puppeteer');
const path = require('path');

const args = process.argv.slice(2);
function getArg(name) {
  const idx = args.indexOf(`--${name}`);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : null;
}

const title = getArg('title') || 'Untitled';
const subtitle = getArg('subtitle') || '';
const tags = (getArg('tags') || '').split(',').filter(Boolean);
const output = getArg('output') || 'featured.png';
const theme = getArg('theme') || 'blue';

const themes = {
  blue:   { bg: 'linear-gradient(135deg, #0f172a 0%, #1e3a5f 40%, #3b82f6 100%)', accent: '#60a5fa' },
  purple: { bg: 'linear-gradient(135deg, #1a1a2e 0%, #4a1942 40%, #8b5cf6 100%)', accent: '#a78bfa' },
  green:  { bg: 'linear-gradient(135deg, #0a2e1a 0%, #134e3a 40%, #10b981 100%)', accent: '#34d399' },
  orange: { bg: 'linear-gradient(135deg, #2d1b0e 0%, #78350f 40%, #f59e0b 100%)', accent: '#fbbf24' },
  red:    { bg: 'linear-gradient(135deg, #1a0a0a 0%, #7f1d1d 40%, #ef4444 100%)', accent: '#f87171' },
  cyan:   { bg: 'linear-gradient(135deg, #0c4a6e 0%, #0e7490 40%, #06b6d4 100%)', accent: '#22d3ee' },
};

const t = themes[theme] || themes.blue;

const tagsHtml = tags.map(tag =>
  `<span style="display:inline-block;padding:6px 16px;background:rgba(255,255,255,0.15);border-radius:20px;font-size:14px;font-weight:600;backdrop-filter:blur(4px);border:1px solid rgba(255,255,255,0.1);">${tag}</span>`
).join(' ');

const html = `<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;width:1200px;height:630px;overflow:hidden;">
<div style="
  width:1200px;height:630px;
  background:${t.bg};
  display:flex;flex-direction:column;justify-content:flex-end;
  padding:60px;box-sizing:border-box;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'PingFang SC','Microsoft YaHei',sans-serif;
  color:white;position:relative;overflow:hidden;
">
  <!-- Decorative elements -->
  <div style="position:absolute;top:-80px;right:-80px;width:400px;height:400px;border-radius:50%;background:rgba(255,255,255,0.03);"></div>
  <div style="position:absolute;top:60px;right:60px;width:200px;height:200px;border-radius:50%;background:rgba(255,255,255,0.05);"></div>
  <div style="position:absolute;bottom:-60px;left:-60px;width:300px;height:300px;border-radius:50%;background:rgba(255,255,255,0.03);"></div>

  <!-- Grid pattern -->
  <div style="position:absolute;inset:0;background-image:radial-gradient(circle at 1px 1px, rgba(255,255,255,0.03) 1px, transparent 0);background-size:40px 40px;"></div>

  <!-- Content -->
  <div style="position:relative;z-index:2;">
    <div style="font-size:56px;font-weight:900;line-height:1.2;margin-bottom:16px;text-shadow:0 4px 20px rgba(0,0,0,0.3);">${title}</div>
    ${subtitle ? `<div style="font-size:24px;font-weight:400;color:rgba(255,255,255,0.8);margin-bottom:24px;line-height:1.4;">${subtitle}</div>` : ''}
    <div style="display:flex;gap:10px;flex-wrap:wrap;">${tagsHtml}</div>
  </div>

  <!-- Bottom accent line -->
  <div style="position:absolute;bottom:0;left:0;right:0;height:4px;background:${t.accent};"></div>
</div>
</body>
</html>`;

(async () => {
  const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 630, deviceScaleFactor: 2 });
  await page.setContent(html, { waitUntil: 'networkidle0' });
  await page.screenshot({ path: output, type: 'png' });
  await browser.close();
  console.log(`Cover generated: ${output} (${title})`);
})();
