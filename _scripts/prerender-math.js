#!/usr/bin/env node
// Render the math in a built site at build time, so readers don't need to run MathJax.
//
// usage: node _scripts/prerender-math.js [site dir, default _site]
//
// Only pages that include _includes/mathjax.html (posts with `math: true`) are touched. Their
// settings are read from the page's #mathjax-config block, the same settings MathJax uses in the
// browser during `make serve`. The math is rendered as HTML and CSS (MathJax's CHTML output, with
// its fonts self-hosted in js/mathjax/), with hidden MathML for screen readers, and the config block
// is removed. Math in handwritten figures goes through js/handwritten-math.js, as in the browser.
// Any TeX error fails the build.

const fs = require('fs');
const path = require('path');
const { mathjax } = require('mathjax-full/js/mathjax.js');
const { TeX } = require('mathjax-full/js/input/tex.js');
const { AllPackages } = require('mathjax-full/js/input/tex/AllPackages.js');
const { CHTML } = require('mathjax-full/js/output/chtml.js');
const { liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js');
const { RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js');
const { AssistiveMmlHandler } = require('mathjax-full/js/a11y/assistive-mml.js');
require('mathjax-full/js/util/entities/all.js'); // HTML entities, so parsing pages needs no async loading
const handwritten = require('../js/handwritten-math.js');

const CONFIG_BLOCK = /<script type="application\/json" id="mathjax-config">([\s\S]*?)<\/script>\s*/;

// Unlike in the browser, an undefined macro or a TeX error must fail the build, not render in red.
const PACKAGES = AllPackages.filter(name => name !== 'noerrors' && name !== 'noundefined');

const adaptor = liteAdaptor();
AssistiveMmlHandler(RegisterHTMLHandler(adaptor));

function htmlFiles(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return htmlFiles(full);
    return entry.name.endsWith('.html') ? [full] : [];
  });
}

function renderPage(html) {
  const config = JSON.parse(html.match(CONFIG_BLOCK)[1]);
  const tex = new TeX({ ...config.tex, packages: PACKAGES });
  tex.postFilters.add(handwritten.filter, handwritten.priority);
  const chtml = new CHTML(config.chtml);
  const doc = mathjax.document(html.replace(CONFIG_BLOCK, ''), { InputJax: tex, OutputJax: chtml });

  // MathJax scales math to the x-height of the surrounding font. Node can't measure it and assumes
  // half the font size, which is right for the body font (Lora); for handwritten figures, use the
  // handwriting font's, as the browser does during `make serve`.
  doc.findMath().compile().getMetrics();
  for (const item of doc.math) handwritten.setFigureMetrics(item, adaptor, chtml);
  doc.render();

  // Node can't measure fonts, so MathJax guessed the width of text in math: drop the guess and let
  // the browser size it from the real font, as the page does in the browser during `make serve`
  // (and the minimum widths of tables, such as aligned derivations, which were computed from those guesses)
  const dropStyle = (tag, property) => {
    for (const node of adaptor.tags(adaptor.body(doc.document), tag)) {
      const style = (adaptor.getAttribute(node, 'style') || '')
        .replace(new RegExp(`(^|;)\\s*${property}:[^;]*;?`), '$1').trim();
      if (style) adaptor.setAttribute(node, 'style', style);
      else adaptor.removeAttribute(node, 'style');
    }
  };
  dropStyle('mjx-utext', 'width');
  dropStyle('mjx-mtable', 'min-width');

  const errors = [];
  for (const item of doc.math) {
    item.root.walkTree(node => {
      if (node.isKind('merror')) {
        errors.push(`${node.attributes.get('data-mjx-error') || 'TeX error'} in: ${item.math.trim()}`);
      }
    });
  }

  adaptor.append(adaptor.head(doc.document), chtml.styleSheet(doc));
  const out = adaptor.doctype(doc.document) + '\n' + adaptor.outerHTML(adaptor.root(doc.document)) + '\n';

  // TeX that MathJax didn't recognize as math (e.g. unbalanced braces) would show up raw on the page
  const prose = adaptor.outerHTML(adaptor.body(doc.document))
    .replace(/<(pre|code|script|style)\b[\s\S]*?<\/\1>/g, '')
    .replace(/<[^>]*>/g, ' ');
  for (const raw of prose.match(/\\[([][^]{0,60}/g) || []) {
    errors.push(`unrendered TeX (unbalanced braces?) at: ${raw.replace(/\s+/g, ' ').trim()}`);
  }

  return { out, count: Array.from(doc.math).length, errors };
}

const siteDir = process.argv[2] || '_site';
let pages = 0;
let equations = 0;
let failed = false;

for (const file of htmlFiles(siteDir)) {
  const html = fs.readFileSync(file, 'utf8');
  if (!CONFIG_BLOCK.test(html)) continue;
  const { out, count, errors } = renderPage(html);
  for (const error of errors) console.error(`${path.relative(siteDir, file)}: ${error}`);
  failed = failed || errors.length > 0;
  fs.writeFileSync(file, out);
  pages += 1;
  equations += count;
}

console.log(`prerender-math: rendered ${equations} equations on ${pages} pages`);
if (failed) {
  console.error('prerender-math: TeX errors found, see above');
  process.exit(1);
}
