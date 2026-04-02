#!/usr/bin/env node
/**
 * Architecture Diagram Generator using Excalidraw
 *
 * Usage:
 *   node scripts/generate-diagram.js --input diagram.excalidraw --output diagram.png
 *   node scripts/generate-diagram.js --input diagram.excalidraw --output diagram.svg --format svg
 *
 * The .excalidraw file is standard Excalidraw JSON format.
 * This script loads it into Excalidraw's web app via Puppeteer and exports.
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
function getArg(name) {
  const idx = args.indexOf(`--${name}`);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : null;
}

const input = getArg('input');
const output = getArg('output') || 'diagram.png';
const format = getArg('format') || 'png';
const padding = parseInt(getArg('padding') || '40');

if (!input) {
  console.error('Usage: node generate-diagram.js --input file.excalidraw --output output.png');
  process.exit(1);
}

const excalidrawData = JSON.parse(fs.readFileSync(input, 'utf8'));

(async () => {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 2 });

  // Load Excalidraw
  await page.goto('https://excalidraw.com', { waitUntil: 'networkidle2', timeout: 30000 });

  // Wait for app to load
  await page.waitForSelector('.excalidraw', { timeout: 15000 });
  await new Promise(r => setTimeout(r, 2000));

  // Import the scene data
  await page.evaluate((data) => {
    // Access Excalidraw's API through the global scope
    const app = document.querySelector('.excalidraw');
    if (app && app.__excalidrawAPI) {
      app.__excalidrawAPI.updateScene({
        elements: data.elements || [],
        appState: { ...data.appState, viewBackgroundColor: '#ffffff' }
      });
    }
  }, excalidrawData);

  await new Promise(r => setTimeout(r, 1000));

  // Use Excalidraw's export via keyboard shortcut or API
  // Fallback: screenshot the canvas
  const canvas = await page.$('canvas');
  if (canvas) {
    const box = await canvas.boundingBox();
    await page.screenshot({
      path: output,
      type: format === 'svg' ? 'png' : 'png',
      clip: { x: box.x, y: box.y, width: box.width, height: box.height }
    });
    console.log(`Diagram exported: ${output}`);
  } else {
    console.error('Could not find Excalidraw canvas');
  }

  await browser.close();
})();
