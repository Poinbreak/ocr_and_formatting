/* ===================================================================
   DigitWin OCR Dashboard — Frontend Logic
   ================================================================ */

(function () {
  'use strict';

  // ------------------------------------------------------------------
  // State
  // ------------------------------------------------------------------
  let models = [];
  let selectedFile = null;
  let currentTextResultId = null;
  let currentJsonResultId = null;
  let rawTextContent = '';

  // ------------------------------------------------------------------
  // DOM refs
  // ------------------------------------------------------------------
  const $id = (id) => document.getElementById(id);

  const fileInput       = $id('fileInput');
  const uploadZone      = $id('uploadZone');
  const imagePreview    = $id('imagePreview');
  const previewImg      = $id('previewImg');
  const modelSelect     = $id('modelSelect');
  const modelInfo       = $id('modelInfo');
  const modelInfoName   = $id('modelInfoName');
  const modelInfoDesc   = $id('modelInfoDesc');
  const promptInput     = $id('promptInput');
  const enhanceToggle   = $id('enhanceToggle');
  const schemaInput     = $id('schemaInput');

  const btnRunOCR       = $id('btnRunOCR');
  const btnRunPipeline  = $id('btnRunPipeline');
  const btnExtractJSON  = $id('btnExtractJSON');
  const btnCopyText     = $id('btnCopyText');
  const btnDownloadText = $id('btnDownloadText');
  const btnCopyJSON     = $id('btnCopyJSON');
  const btnDownloadJSON = $id('btnDownloadJSON');

  const textOutput        = $id('textOutput');
  const textToolbar       = $id('textToolbar');
  const textStats         = $id('textStats');
  const textLoadingOverlay= $id('textLoadingOverlay');
  const textLoadingText   = $id('textLoadingText');

  const jsonOutput        = $id('jsonOutput');
  const jsonToolbar       = $id('jsonToolbar');
  const jsonStats         = $id('jsonStats');
  const jsonLoadingOverlay= $id('jsonLoadingOverlay');
  const jsonLoadingText   = $id('jsonLoadingText');

  const pipelineProgress  = $id('pipelineProgress');
  const toastContainer    = $id('toastContainer');

  const ollamaStatusDot   = $id('ollamaStatusDot');
  const ollamaStatusText  = $id('ollamaStatusText');
  const modelStatusDot    = $id('modelStatusDot');
  const modelStatusText   = $id('modelStatusText');

  // ------------------------------------------------------------------
  // Initialize
  // ------------------------------------------------------------------
  async function init() {
    await fetchModels();
    checkOllama();
    wireEvents();
  }

  async function fetchModels() {
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      models = data.models;
      schemaInput.value = data.default_schema || '';

      // Populate select
      modelSelect.innerHTML = '';
      const groups = {};
      models.forEach((m) => {
        if (!groups[m.group]) groups[m.group] = [];
        groups[m.group].push(m);
      });

      for (const [group, items] of Object.entries(groups)) {
        const optgroup = document.createElement('optgroup');
        optgroup.label = group;
        items.forEach((m) => {
          const opt = document.createElement('option');
          opt.value = m.key;
          if (m.available === false) {
            opt.textContent = m.name + ' (not installed)';
            opt.disabled = true;
          } else {
            opt.textContent = m.name;
          }
          optgroup.appendChild(opt);
        });
        modelSelect.appendChild(optgroup);
      }

      // Select first
      if (models.length > 0) {
        modelSelect.value = models[0].key;
        onModelChange();
      }

      // Update loaded indicator
      if (data.loaded) {
        modelStatusDot.className = 'status-dot online';
        const loadedModel = models.find((m) => m.key === data.loaded);
        modelStatusText.textContent = loadedModel ? loadedModel.name : data.loaded;
      }
    } catch (e) {
      toast('Failed to load models: ' + e.message, 'error');
    }
  }

  async function checkOllama() {
    try {
      const res = await fetch('/api/ollama/status');
      const data = await res.json();
      ollamaStatusDot.className = data.online ? 'status-dot online' : 'status-dot offline';
      ollamaStatusText.textContent = data.online ? 'Ollama Online' : 'Ollama Offline';
    } catch {
      ollamaStatusDot.className = 'status-dot offline';
      ollamaStatusText.textContent = 'Ollama Offline';
    }
  }

  // ------------------------------------------------------------------
  // Events
  // ------------------------------------------------------------------
  function wireEvents() {
    // File input
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadZone.classList.add('drag-over');
    });
    uploadZone.addEventListener('dragleave', () => {
      uploadZone.classList.remove('drag-over');
    });
    uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadZone.classList.remove('drag-over');
      if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelect();
      }
    });

    // Model change
    modelSelect.addEventListener('change', onModelChange);

    // Buttons
    btnRunOCR.addEventListener('click', runOCR);
    btnRunPipeline.addEventListener('click', runPipeline);
    btnExtractJSON.addEventListener('click', runExtract);
    btnCopyText.addEventListener('click', () => copyToClipboard(rawTextContent, 'Text'));
    btnDownloadText.addEventListener('click', () => downloadResult('txt', currentTextResultId));
    btnCopyJSON.addEventListener('click', () => copyToClipboard(jsonOutput.textContent, 'JSON'));
    btnDownloadJSON.addEventListener('click', () => downloadResult('json', currentJsonResultId));
  }

  function handleFileSelect() {
    const file = fileInput.files[0];
    if (!file) return;
    selectedFile = file;
    uploadZone.classList.add('has-file');

    // Preview
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      imagePreview.classList.add('visible');
    };
    reader.readAsDataURL(file);

    updateButtons();
  }

  function onModelChange() {
    const key = modelSelect.value;
    const meta = models.find((m) => m.key === key);
    if (!meta) return;

    promptInput.value = meta.default_prompt;

    modelInfoName.textContent = meta.name;
    modelInfoDesc.textContent = meta.description;
    modelInfo.classList.remove('hidden');

    updateButtons();
  }

  function updateButtons() {
    const hasFile = selectedFile !== null;
    const hasModel = modelSelect.value !== '';
    btnRunOCR.disabled = !(hasFile && hasModel);
    btnRunPipeline.disabled = !(hasFile && hasModel);
  }

  // ------------------------------------------------------------------
  // OCR
  // ------------------------------------------------------------------
  async function runOCR() {
    if (!selectedFile || !modelSelect.value) return;

    showLoading(textLoadingOverlay, textLoadingText, 'Loading model & running OCR…');
    btnRunOCR.disabled = true;

    const form = new FormData();
    form.append('image', selectedFile);
    form.append('model', modelSelect.value);
    form.append('prompt', promptInput.value);
    form.append('enhance', enhanceToggle.checked ? 'true' : 'false');

    try {
      const res = await fetch('/api/ocr', { method: 'POST', body: form });
      const data = await res.json();

      if (data.error) {
        toast(data.error, 'error');
        return;
      }

      displayText(data.text, data);
      currentTextResultId = data.result_id;
      rawTextContent = data.text;

      // Update model status
      modelStatusDot.className = 'status-dot online';
      modelStatusText.textContent = data.model;

      btnExtractJSON.disabled = false;
      toast(`OCR complete — ${data.char_count} chars`, 'success');
    } catch (e) {
      toast('OCR failed: ' + e.message, 'error');
    } finally {
      hideLoading(textLoadingOverlay);
      btnRunOCR.disabled = false;
    }
  }

  // ------------------------------------------------------------------
  // JSON Extraction
  // ------------------------------------------------------------------
  async function runExtract() {
    if (!rawTextContent) return;

    showLoading(jsonLoadingOverlay, jsonLoadingText, 'Running NuExtract…');
    btnExtractJSON.disabled = true;

    try {
      const res = await fetch('/api/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: rawTextContent,
          schema: schemaInput.value,
        }),
      });
      const data = await res.json();

      if (data.error) {
        toast(data.error, 'error');
        return;
      }

      displayJSON(data.json);
      currentJsonResultId = data.result_id;
      toast('JSON extraction complete', 'success');
    } catch (e) {
      toast('Extraction failed: ' + e.message, 'error');
    } finally {
      hideLoading(jsonLoadingOverlay);
      btnExtractJSON.disabled = false;
    }
  }

  // ------------------------------------------------------------------
  // Full Pipeline
  // ------------------------------------------------------------------
  async function runPipeline() {
    if (!selectedFile || !modelSelect.value) return;

    // Show pipeline progress
    pipelineProgress.classList.add('visible');
    setPipelineStep(1);
    btnRunPipeline.disabled = true;
    btnRunOCR.disabled = true;

    showLoading(textLoadingOverlay, textLoadingText, 'Running full pipeline…');
    showLoading(jsonLoadingOverlay, jsonLoadingText, 'Waiting for OCR…');

    const form = new FormData();
    form.append('image', selectedFile);
    form.append('model', modelSelect.value);
    form.append('prompt', promptInput.value);
    form.append('schema', schemaInput.value);
    form.append('enhance', enhanceToggle.checked ? 'true' : 'false');

    // Simulate progress (since the backend is synchronous)
    const progressTimer = simulatePipelineProgress();

    try {
      const res = await fetch('/api/pipeline', { method: 'POST', body: form });
      const data = await res.json();

      clearInterval(progressTimer);

      if (data.error) {
        setPipelineStepError();
        toast(data.error, 'error');
        return;
      }

      // All steps done
      setPipelineStep(5); // all done

      // Display results
      displayText(data.text, data);
      currentTextResultId = data.text_id;
      rawTextContent = data.text;

      displayJSON(data.json);
      currentJsonResultId = data.json_id;

      // Update model status
      modelStatusDot.className = 'status-dot online';
      modelStatusText.textContent = data.model;

      btnExtractJSON.disabled = false;
      toast('Pipeline complete!', 'success');
    } catch (e) {
      clearInterval(progressTimer);
      setPipelineStepError();
      toast('Pipeline failed: ' + e.message, 'error');
    } finally {
      hideLoading(textLoadingOverlay);
      hideLoading(jsonLoadingOverlay);
      btnRunPipeline.disabled = false;
      btnRunOCR.disabled = false;
    }
  }

  function simulatePipelineProgress() {
    let step = 1;
    return setInterval(() => {
      step++;
      if (step <= 4) setPipelineStep(step);
    }, 8000); // advance every 8s as an estimate
  }

  function setPipelineStep(n) {
    for (let i = 1; i <= 4; i++) {
      const el = $id('pStep' + i);
      el.classList.remove('active', 'done', 'error');
      if (i < n) el.classList.add('done');
      else if (i === n && n <= 4) el.classList.add('active');
    }
    for (let i = 1; i <= 3; i++) {
      const conn = $id('pConn' + i);
      conn.classList.toggle('done', i < n);
    }
  }

  function setPipelineStepError() {
    // Mark the currently active step as error
    for (let i = 1; i <= 4; i++) {
      const el = $id('pStep' + i);
      if (el.classList.contains('active')) {
        el.classList.remove('active');
        el.classList.add('error');
        break;
      }
    }
  }

  // ------------------------------------------------------------------
  // Display helpers
  // ------------------------------------------------------------------
  function displayText(text, meta) {
    textToolbar.classList.remove('hidden');
    textStats.innerHTML = `
      <span>${meta.char_count || text.length} chars</span>
      <span>${meta.word_count || text.split(/\s+/).length} words</span>
    `;
    textOutput.innerHTML = `<div class="output-content">${escapeHTML(text)}</div>`;
  }

  function displayJSON(jsonStr) {
    jsonToolbar.classList.remove('hidden');

    // Try to compute stats
    try {
      const obj = JSON.parse(jsonStr);
      const keys = Object.keys(obj).length;
      jsonStats.innerHTML = `<span>${keys} top-level keys</span><span>${jsonStr.length} chars</span>`;
    } catch {
      jsonStats.innerHTML = `<span>${jsonStr.length} chars</span>`;
    }

    jsonOutput.innerHTML = `<div class="output-content">${highlightJSON(jsonStr)}</div>`;
  }

  // ------------------------------------------------------------------
  // JSON syntax highlighting
  // ------------------------------------------------------------------
  function highlightJSON(str) {
    // Escape HTML first
    let html = escapeHTML(str);

    // Highlight keys "key":
    html = html.replace(
      /(&quot;)(.*?)(&quot;)\s*:/g,
      '<span class="json-key">$1$2$3</span>:'
    );

    // Highlight string values (after colon)
    html = html.replace(
      /:\s*(&quot;)(.*?)(&quot;)/g,
      ': <span class="json-string">$1$2$3</span>'
    );

    // Highlight numbers
    html = html.replace(
      /:\s*(\d+\.?\d*)/g,
      ': <span class="json-number">$1</span>'
    );

    // Highlight booleans
    html = html.replace(
      /:\s*(true|false)/g,
      ': <span class="json-bool">$1</span>'
    );

    // Highlight null
    html = html.replace(
      /:\s*(null)/g,
      ': <span class="json-null">$1</span>'
    );

    // Highlight brackets
    html = html.replace(
      /([{}\[\]])/g,
      '<span class="json-bracket">$1</span>'
    );

    return html;
  }

  function escapeHTML(str) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  // ------------------------------------------------------------------
  // Loading overlay
  // ------------------------------------------------------------------
  function showLoading(overlay, textEl, msg) {
    if (textEl) textEl.textContent = msg;
    overlay.classList.add('visible');
  }

  function hideLoading(overlay) {
    overlay.classList.remove('visible');
  }

  // ------------------------------------------------------------------
  // Copy & Download
  // ------------------------------------------------------------------
  async function copyToClipboard(text, label) {
    try {
      await navigator.clipboard.writeText(text);
      toast(`${label} copied to clipboard`, 'success');
    } catch {
      toast('Failed to copy', 'error');
    }
  }

  function downloadResult(type, resultId) {
    if (!resultId) {
      toast('No result to download', 'error');
      return;
    }
    window.open(`/api/download/${type}/${resultId}`, '_blank');
  }

  // ------------------------------------------------------------------
  // Toast Notifications
  // ------------------------------------------------------------------
  function toast(message, type = 'info') {
    const icons = { success: '✅', error: '❌', info: 'ℹ️' };

    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.innerHTML = `
      <span class="toast-icon">${icons[type] || icons.info}</span>
      <span>${escapeHTML(message)}</span>
      <button class="toast-close" onclick="this.parentElement.classList.add('exit');setTimeout(()=>this.parentElement.remove(),300)">✕</button>
    `;
    toastContainer.appendChild(el);

    // Auto-remove after 5s
    setTimeout(() => {
      if (el.parentElement) {
        el.classList.add('exit');
        setTimeout(() => el.remove(), 300);
      }
    }, 5000);
  }

  // ------------------------------------------------------------------
  // Boot
  // ------------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', init);
})();
