#!/usr/bin/env node
/**
 * Blog Cover Generator V2 - With architecture diagram content
 *
 * Usage:
 *   node scripts/generate-cover-v2.js --config cover-config.json --output featured.png
 *
 * Or pass inline:
 *   node scripts/generate-cover-v2.js --title "SORT" --subtitle "..." --diagram "html string" --output featured.png --theme blue
 */

const puppeteer = require('puppeteer');

const args = process.argv.slice(2);
function getArg(name) {
  const idx = args.indexOf(`--${name}`);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : null;
}

const title = getArg('title') || 'Untitled';
const subtitle = getArg('subtitle') || '';
const diagramHtml = getArg('diagram') || '';
const output = getArg('output') || 'featured.png';
const theme = getArg('theme') || 'blue';

const themes = {
  blue:   { bg: '#ffffff', accent: '#3b82f6', accent2: '#2563eb', gradient: '#ffffff', textColor: '#111827' },
  purple: { bg: '#ffffff', accent: '#8b5cf6', accent2: '#7c3aed', gradient: '#ffffff', textColor: '#111827' },
  green:  { bg: '#ffffff', accent: '#10b981', accent2: '#059669', gradient: '#ffffff', textColor: '#111827' },
  orange: { bg: '#ffffff', accent: '#f59e0b', accent2: '#d97706', gradient: '#ffffff', textColor: '#111827' },
  red:    { bg: '#ffffff', accent: '#ef4444', accent2: '#dc2626', gradient: '#ffffff', textColor: '#111827' },
  cyan:   { bg: '#ffffff', accent: '#06b6d4', accent2: '#0891b2', gradient: '#ffffff', textColor: '#111827' },
};

const t = themes[theme] || themes.blue;

function buildBox(label, color) {
  return `<div style="padding:8px 16px;background:${color || t.accent};border-radius:8px;font-size:13px;font-weight:700;color:white;text-align:center;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,0.3);">${label}</div>`;
}

function buildArrow() {
  return `<div style="color:${t.accent2};font-size:20px;line-height:1;">→</div>`;
}

function buildArrowDown() {
  return `<div style="color:${t.accent2};font-size:20px;line-height:1;text-align:center;">↓</div>`;
}

// If no custom diagram HTML, build from title keywords
let diagram = diagramHtml;
if (!diagram) {
  // Auto-generate a simple flow based on common patterns
  diagram = `<div style="display:flex;flex-direction:column;gap:8px;align-items:center;">
    ${buildBox('Input', 'rgba(255,255,255,0.1)')}
    ${buildArrowDown()}
    ${buildBox(title, t.accent)}
    ${buildArrowDown()}
    ${buildBox('Output', 'rgba(255,255,255,0.1)')}
  </div>`;
}

const html = `<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;width:1200px;height:630px;overflow:hidden;">
<div style="
  width:1200px;height:630px;
  background:${t.gradient};
  display:flex;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;
  color:${t.textColor};position:relative;overflow:hidden;
">
  <!-- Subtle dot grid -->
  <div style="position:absolute;inset:0;background-image:radial-gradient(circle at 1px 1px, rgba(0,0,0,0.04) 1px, transparent 0);background-size:32px 32px;"></div>

  <!-- Left: Architecture diagram -->
  <div style="flex:1;display:flex;align-items:center;justify-content:center;padding:40px;position:relative;z-index:2;">
    ${diagram}
  </div>

  <!-- Vertical divider -->
  <div style="width:1px;background:rgba(0,0,0,0.08);margin:60px 0;"></div>

  <!-- Right: Title + subtitle -->
  <div style="flex:1;display:flex;flex-direction:column;justify-content:center;padding:50px;position:relative;z-index:2;">
    <div style="font-size:48px;font-weight:900;line-height:1.15;margin-bottom:16px;letter-spacing:-1px;color:#111827;">${title}</div>
    <div style="font-size:20px;color:#6b7280;line-height:1.5;">${subtitle}</div>
    <div style="margin-top:24px;height:3px;width:60px;background:${t.accent};border-radius:2px;"></div>
  </div>
</div>
</body></html>`;

(async () => {
  const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 630, deviceScaleFactor: 2 });
  await page.setContent(html, { waitUntil: 'networkidle0' });
  await page.screenshot({ path: output, type: 'png' });
  await browser.close();
  console.log(`Cover generated: ${output}`);
})();
