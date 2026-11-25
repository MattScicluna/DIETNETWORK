// Typeset the math of the page that is shown, once. Both MathJax's start and Material's
// page event ask for it, in either order; and clearing MathJax's cache on a page that is
// already typeset would leave its formulas without glyphs.
function typesetMath() {
  if (!window.MathJax.typesetPromise) return; // not loaded yet: its start will call this
  if (document.querySelector(".arithmatex mjx-container")) return;
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
}

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
  startup: {
    typeset: false,
    ready: () => {
      MathJax.startup.defaultReady();
      MathJax.startup.promise.then(typesetMath);
    },
  },
};

// navigation.instant swaps pages without a reload.
document$.subscribe(typesetMath);
