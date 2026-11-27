/*
 * Handwritten figures written in TeX.
 *
 * Inside an element with class "hw" (see css/handwritten.css), the math is drawn in the surrounding
 * handwriting font, while MathJax still does the layout: superscripts, subscripts, aligned "=" signs,
 * matrices and so on. Colors from \class{math-...}{...} are kept.
 *
 * Used by the browser during `make serve` (see _includes/mathjax.html) and by
 * _scripts/prerender-math.js during `make build`, so that both render figures the same way:
 * - `filter` is a MathJax TeX post-filter: it swaps each identifier, number and operator in the parsed
 *   formula for a text node, which MathJax draws in the surrounding font;
 * - `setFigureMetrics` tells MathJax the handwriting font's x-height, which it scales the layout to.
 */
(function (root) {
  // operators get a class so css/handwritten.css can space them like the drawings
  var OPERATOR_CLASS = {
    '=': 'hw-rel', '≡': 'hw-rel',
    '+': 'hw-bin', '−': 'hw-bin', '-': 'hw-bin', '⋅': 'hw-bin', '·': 'hw-bin',
    ',': 'hw-punct'
  };
  // the handwriting font has the middle dot but not the math dot operator that \cdot produces
  var REPLACE = { '⋅': '·' };

  // Height of the handwriting font's lowercase letters, as a fraction of its size. MathJax scales the
  // layout to the x-height of the surrounding font; the 0.365 that browsers read from Indie Flower's
  // font file is smaller than its letters (0.48), which leaves too little room in brackets and rows.
  var X_HEIGHT = 0.48;

  // more room between the rows of derivations (\begin{aligned}) than MathJax's default, like the drawings
  var DERIVATION_ROW_SPACING = '0.9em';

  function inFigure(node, adaptor) {
    for (var n = node; n; n = adaptor.parent(n)) {
      // only elements have classes: skip text and comment nodes ("#text", "#comment") and the
      // document itself (its kind is "" in the browser)
      var kind = adaptor.kind(n);
      if (kind && kind.charAt(0) !== '#' && adaptor.hasClass(n, 'hw')) return true;
    }
    return false;
  }

  // Brackets that grow with their contents (\left[, matrices) and arrows are drawn by MathJax, which
  // builds them from pieces of its own font: as handwritten text they couldn't stretch.
  function isStretchyOrArrow(mo) {
    var text = mo.getText();
    var code = text.length === 1 ? text.charCodeAt(0) : 0;
    var arrow = (code >= 0x2190 && code <= 0x21FF) || (code >= 0x27F0 && code <= 0x27FF) || (code >= 0x2900 && code <= 0x297F);
    return arrow || mo.attributes.getExplicit('stretchy') === true || mo.attributes.getExplicit('fence') === true;
  }

  function filter(args) {
    var math = args.math, data = args.data, adaptor = args.document.adaptor;
    if (!math.start || !math.start.node || !inFigure(math.start.node, adaptor)) return;
    var tokens = [];
    data.root.walkTree(function (node) {
      // spacing set here, unlike in CSS, is accounted for by MathJax when it sizes brackets;
      // derivations are display style, matrices and arrays aren't, so they keep their spacing
      var display = node.isKind('mtable') && node.attributes.getExplicit('displaystyle');
      if (display === true || display === 'true') node.attributes.set('rowspacing', DERIVATION_ROW_SPACING);
      if (node.isKind('mi') || node.isKind('mn')) tokens.push(node);
      else if (node.isKind('mo') && !isStretchyOrArrow(node)) tokens.push(node);
    });
    tokens.forEach(function (token) {
      var text = token.getText();
      if (!text) return;
      var mtext = data.nodeFactory.create('token', 'mtext', {}, REPLACE[text] || text);
      // keep the symbol's own attributes, such as the color class that \class{math-green}{x} puts
      // on a single symbol, but not its font variant, so it's drawn in the handwriting font
      var attributes = token.attributes.getAllAttributes();
      Object.keys(attributes).forEach(function (name) {
        if (name !== 'mathvariant') mtext.attributes.set(name, attributes[name]);
      });
      var classes = [attributes['class'], token.isKind('mo') && OPERATOR_CLASS[text]].filter(Boolean);
      if (classes.length) mtext.attributes.set('class', classes.join(' '));
      token.parent.replaceChild(mtext, token);
    });
  }

  // For math in a figure, use the handwriting font's x-height instead of the one MathJax measured or,
  // at build time, assumed. Call after MathJax has computed the metrics and before it typesets.
  function setFigureMetrics(item, adaptor, outputJax) {
    if (!item.start || !item.start.node || !inFigure(item.start.node, adaptor)) return;
    var m = item.metrics;
    item.setMetrics(m.em, m.em * X_HEIGHT, m.containerWidth, m.lineWidth, X_HEIGHT / outputJax.font.params.x_height);
  }

  var api = {
    filter: filter,
    // Run before MathJax's own clean-up filters (priorities -6 to -1), which among other things mark
    // each node with its script level: that's what makes text in superscripts and subscripts smaller.
    priority: -10,
    setFigureMetrics: setFigureMetrics
  };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.HandwrittenMath = api;
})(this);
