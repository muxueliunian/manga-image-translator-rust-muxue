const i18n = {
  zh: {
    eyebrow: "WebView 工作台",
    title: "漫画图片翻译",
    stepInput: "选择素材",
    stepInputHint: "图片或文件夹",
    stepModel: "配置模型",
    stepModelHint: "OCR 与翻译器",
    stepRun: "翻译导出",
    stepRunHint: "查看结果与日志",
    inputKicker: "输入",
    inputTitle: "选择图片或文件夹",
    pickImages: "选择图片",
    pickFolder: "选择文件夹",
    clearInputs: "清空",
    dropTitle: "从这里开始：选择漫画页、单张图或整本文件夹",
    dropHint: "支持多选图片或选择整本文件夹，翻译完成后会先生成可预览结果，再按需导出。",
    outputDir: "导出目录",
    outputFormat: "输出格式",
    textDirection: "文字方向",
    textDirectionAuto: "自动",
    textDirectionHorizontal: "横排",
    textDirectionVertical: "竖排",
    configKicker: "模型",
    configTitle: "翻译配置",
    reloadDefaults: "默认值",
    loadConfig: "加载配置",
    saveConfig: "保存配置",
    translator: "翻译器",
    targetLang: "目标语言",
    provider: "模型供应商",
    baseUrl: "Base URL",
    apiKey: "API Key",
    modelName: "模型名称",
    temperature: "Temperature",
    topP: "Top P",
    systemPrompt: "System Prompt",
    userPrompt: "User Prompt Template",
    advancedJson: "高级 JSON 配置",
    readyTitle: "准备就绪",
    readyText: "选择输入与输出目录后即可开始。",
    start: "开始翻译",
    resultKicker: "结果",
    resultTitle: "预览与导出",
    resultEmpty: "暂无结果。完成翻译后可预览图片，并选择导出到指定目录。",
    selectAllResults: "全选",
    deselectAllResults: "取消全选",
    exportSelected: "导出选中",
    preview: "预览",
    exported: "已导出",
    exportNeedSelection: "请先勾选要导出的结果。",
    exportNeedDir: "请先选择导出目录。",
    remove: "删除",
    logKicker: "日志",
    logTitle: "运行记录",
    clearLog: "清空",
    noInput: "未选择输入",
    selected: "已选择",
    folderSelected: "已选择文件夹",
    outputSelected: "导出目录已设置",
    defaultsLoaded: "默认配置已加载",
    configLoaded: "配置已加载",
    configSaved: "配置已保存",
    starting: "已发送任务",
    backendPending: "正在执行翻译任务",
    progressIdle: "等待任务",
    progressPreparing: "正在准备模型",
    progressRunning: "正在处理",
    progressDone: "处理完成",
    jsonError: "JSON 配置格式错误",
  },
  en: {
    eyebrow: "WebView Workspace",
    title: "Manga Image Translator",
    stepInput: "Choose Source",
    stepInputHint: "Images or folder",
    stepModel: "Configure Models",
    stepModelHint: "OCR and translator",
    stepRun: "Translate Export",
    stepRunHint: "Results and logs",
    inputKicker: "Input",
    inputTitle: "Choose Images or Folder",
    pickImages: "Choose Images",
    pickFolder: "Choose Folder",
    clearInputs: "Clear",
    dropTitle: "Start here: choose manga pages, one image, or a whole folder",
    dropHint: "Select images or a folder. Finished pages are cached for preview, then exported on demand.",
    outputDir: "Export Directory",
    outputFormat: "Output Format",
    textDirection: "Text Direction",
    textDirectionAuto: "Auto",
    textDirectionHorizontal: "Horizontal",
    textDirectionVertical: "Vertical",
    configKicker: "Models",
    configTitle: "Translation Config",
    reloadDefaults: "Defaults",
    loadConfig: "Load Config",
    saveConfig: "Save Config",
    translator: "Translator",
    targetLang: "Target Language",
    provider: "Provider",
    baseUrl: "Base URL",
    apiKey: "API Key",
    modelName: "Model Name",
    temperature: "Temperature",
    topP: "Top P",
    systemPrompt: "System Prompt",
    userPrompt: "User Prompt Template",
    advancedJson: "Advanced JSON Config",
    readyTitle: "Ready",
    readyText: "Choose input and output directory to begin.",
    start: "Start Translating",
    resultKicker: "Results",
    resultTitle: "Preview and Export",
    resultEmpty: "No results yet. Finished images can be previewed and exported after translation.",
    selectAllResults: "Select All",
    deselectAllResults: "Deselect All",
    exportSelected: "Export Selected",
    preview: "Preview",
    exported: "Exported",
    exportNeedSelection: "Select at least one result first.",
    exportNeedDir: "Choose an export directory first.",
    remove: "Remove",
    logKicker: "Logs",
    logTitle: "Run Log",
    clearLog: "Clear",
    noInput: "No input selected",
    selected: "selected",
    folderSelected: "Folder selected",
    outputSelected: "Output directory set",
    defaultsLoaded: "Default settings loaded",
    configLoaded: "Config loaded",
    configSaved: "Config saved",
    starting: "Job sent",
    backendPending: "Translation is running",
    progressIdle: "Idle",
    progressPreparing: "Preparing models",
    progressRunning: "Processing",
    progressDone: "Done",
    jsonError: "Invalid JSON settings",
  },
};

const providerBaseUrls = {
  OpenAI: "https://api.openai.com/v1",
  DeepSeek: "https://api.deepseek.com/v1",
  OpenRouter: "https://openrouter.ai/api/v1",
  SiliconFlow: "https://api.siliconflow.cn/v1",
  DashScope: "https://dashscope.aliyuncs.com/compatible-mode/v1",
  Moonshot: "https://api.moonshot.cn/v1",
  Zhipu: "https://open.bigmodel.cn/api/paas/v4",
};

const state = {
  lang: localStorage.getItem("mitWebviewLang") || "zh",
  inputPaths: [],
  outputDir: "",
  results: [],
  selectedResults: new Set(),
  settings: null,
  requestId: 0,
  pending: new Map(),
};

const els = {
  langToggle: document.getElementById("langToggle"),
  backendBadge: document.getElementById("backendBadge"),
  pickImages: document.getElementById("pickImages"),
  pickFolder: document.getElementById("pickFolder"),
  clearInputs: document.getElementById("clearInputs"),
  pickOutputDir: document.getElementById("pickOutputDir"),
  outputDir: document.getElementById("outputDir"),
  outputFormat: document.getElementById("outputFormat"),
  textDirection: document.getElementById("textDirection"),
  inputList: document.getElementById("inputList"),
  translator: document.getElementById("translator"),
  targetLang: document.getElementById("targetLang"),
  provider: document.getElementById("provider"),
  baseUrl: document.getElementById("baseUrl"),
  apiKey: document.getElementById("apiKey"),
  modelName: document.getElementById("modelName"),
  temperature: document.getElementById("temperature"),
  topP: document.getElementById("topP"),
  systemPrompt: document.getElementById("systemPrompt"),
  userPrompt: document.getElementById("userPrompt"),
  settingsJson: document.getElementById("settingsJson"),
  reloadDefaults: document.getElementById("reloadDefaults"),
  loadConfig: document.getElementById("loadConfig"),
  saveConfig: document.getElementById("saveConfig"),
  startTranslation: document.getElementById("startTranslation"),
  statusTitle: document.getElementById("statusTitle"),
  statusText: document.getElementById("statusText"),
  progressBar: document.getElementById("progressBar"),
  progressLabel: document.getElementById("progressLabel"),
  selectAllResults: document.getElementById("selectAllResults"),
  exportSelected: document.getElementById("exportSelected"),
  results: document.getElementById("results"),
  logList: document.getElementById("logList"),
  clearLog: document.getElementById("clearLog"),
};

window.MIT_BRIDGE = {
  resolve(response) {
    const pending = state.pending.get(response.id);
    if (!pending) return;
    state.pending.delete(response.id);
    if (response.ok) {
      pending.resolve(response.data);
    } else {
      pending.reject(new Error(response.error || "IPC request failed"));
    }
  },
  emit(name, payload) {
    if (name === "log") {
      addLog(payload.level || "info", payload.message || "");
    } else if (name === "progress") {
      updateProgress(payload || {});
    }
  },
};

function t(key) {
  return i18n[state.lang][key] || key;
}

function applyLang() {
  document.documentElement.lang = state.lang === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  els.langToggle.textContent = state.lang === "zh" ? "English" : "中文";
  if (!state.inputPaths.length) renderInputList();
  renderResults();
}

function invoke(kind, payload = {}) {
  const id = `req_${Date.now()}_${++state.requestId}`;
  const message = JSON.stringify({ id, kind, payload });
  return new Promise((resolve, reject) => {
    state.pending.set(id, { resolve, reject });
    const ipc =
      window.ipc && typeof window.ipc.postMessage === "function"
        ? window.ipc
        : window.chrome?.webview && typeof window.chrome.webview.postMessage === "function"
          ? window.chrome.webview
          : null;
    if (!ipc) {
      state.pending.delete(id);
      reject(new Error("WebView IPC bridge is not available."));
      return;
    }
    ipc.postMessage(message);
  });
}

function addLog(level, message) {
  const entry = document.createElement("div");
  entry.className = "log-entry";
  entry.dataset.level = level;
  entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  els.logList.prepend(entry);
}

function setStatus(title, text) {
  els.statusTitle.textContent = title;
  els.statusText.textContent = text;
}

function updateProgress(payload) {
  const current = Number(payload.current ?? 0);
  const total = Number(payload.total ?? 0);
  const percent =
    typeof payload.percent === "number"
      ? payload.percent
      : total > 0
        ? Math.round((current / total) * 100)
        : 0;
  const clamped = Math.max(0, Math.min(100, percent));
  els.progressBar.style.width = `${clamped}%`;
  const message = payload.message || t("progressIdle");
  els.progressLabel.textContent =
    total > 0 ? `${message} · ${current}/${total} · ${clamped}%` : message;
}

function renderInputList() {
  els.inputList.innerHTML = "";
  if (!state.inputPaths.length) {
    const empty = document.createElement("div");
    empty.className = "path-item";
    empty.textContent = t("noInput");
    els.inputList.append(empty);
    return;
  }

  state.inputPaths.forEach((path, index) => {
    const item = document.createElement("div");
    item.className = "path-item";
    const name = path.split(/[\\/]/).filter(Boolean).pop() || path;
    item.innerHTML = `
      <div class="path-text">
        <strong title="${escapeHtml(path)}">${escapeHtml(name)}</strong>
        <span>${escapeHtml(path)}</span>
      </div>
      <button class="tiny-button" type="button" data-remove-input="${index}">${t("remove")}</button>
    `;
    els.inputList.append(item);
  });
}

function removeInput(index) {
  state.inputPaths.splice(index, 1);
  renderInputList();
  setStatus(t("selected"), `${state.inputPaths.length} ${t("selected")}`);
}

function clearInputs() {
  state.inputPaths = [];
  renderInputList();
  setStatus(t("readyTitle"), t("readyText"));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function chooseImages() {
  try {
    setStatus(t("starting"), "正在打开图片选择窗口...");
    const data = await invoke("pickImages");
    state.inputPaths = data.paths || [];
    renderInputList();
    setStatus(t("selected"), `${state.inputPaths.length} ${t("selected")}`);
    addLog("info", `${t("selected")}: ${state.inputPaths.length}`);
  } catch (err) {
    addLog("error", err.message);
  }
}

async function chooseFolder() {
  try {
    setStatus(t("starting"), "正在打开文件夹选择窗口...");
    const data = await invoke("pickFolder");
    state.inputPaths = data.paths || [];
    renderInputList();
    setStatus(t("folderSelected"), state.inputPaths[0] || t("noInput"));
    addLog("info", `${t("folderSelected")}: ${state.inputPaths[0] || "-"}`);
  } catch (err) {
    addLog("error", err.message);
  }
}

async function chooseOutputDir() {
  try {
    setStatus(t("starting"), "正在打开输出目录选择窗口...");
    const data = await invoke("pickOutputDir");
    state.outputDir = (data.paths || [])[0] || "";
    els.outputDir.value = state.outputDir;
    if (state.outputDir) {
      setStatus(t("outputSelected"), state.outputDir);
      addLog("info", `${t("outputSelected")}: ${state.outputDir}`);
    }
  } catch (err) {
    addLog("error", err.message);
  }
}

async function loadDefaults() {
  try {
    const defaults = await invoke("defaults");
    applySettings(defaults);
    addLog("success", t("defaultsLoaded"));
  } catch (err) {
    addLog("error", err.message);
  }
}

async function loadConfig() {
  try {
    const config = await invoke("loadConfig");
    applySettings(config);
    addLog("success", t("configLoaded"));
  } catch (err) {
    addLog("error", err.message);
  }
}

async function saveConfig() {
  try {
    const settings = patchSettingsFromControls();
    const result = await invoke("saveConfig", { settings });
    applySettings(settings);
    addLog("success", `${t("configSaved")}: ${result.path || "config/app.json"}`);
    setStatus(t("configSaved"), result.path || "config/app.json");
  } catch (err) {
    setStatus(t("jsonError"), err.message);
    addLog("error", err.message);
  }
}

function applySettings(settings) {
  state.settings = settings || {};
  els.settingsJson.value = JSON.stringify(state.settings, null, 2);
  syncControlsFromSettings();
}

function syncControlsFromSettings() {
  const cfg = state.settings || {};
  const translation = cfg.translator?.target?.translator ? cfg.translator.target : null;
  const openai = cfg.translator?.openai_compatible || {};
  const render = cfg.render || {};
  els.translator.value = translation?.translator || "Sugoi";
  els.targetLang.value = translation?.target || "en";
  els.textDirection.value = render.text_direction
    ? String(render.text_direction).toLowerCase()
    : "auto";
  els.provider.value = openai.provider_preset || "Custom";
  els.baseUrl.value = openai.base_url || "";
  els.apiKey.value = openai.api_key || "";
  els.modelName.value = openai.model || "";
  els.temperature.value = openai.temperature ?? "";
  els.topP.value = openai.top_p ?? "";
  els.systemPrompt.value = openai.system_prompt || "";
  els.userPrompt.value = openai.user_prompt_template || "";
}

function patchSettingsFromControls() {
  const cfg = JSON.parse(els.settingsJson.value || "{}");
  cfg.translator = cfg.translator || {};
  cfg.translator.target = cfg.translator.target || {};
  if (
    els.provider.value !== "Custom" ||
    els.baseUrl.value.trim() ||
    els.apiKey.value.trim()
  ) {
    els.translator.value = "OpenAICompatible";
  }
  cfg.translator.target.translator = els.translator.value;
  cfg.translator.target.target = els.targetLang.value;
  cfg.translator.openai_compatible = cfg.translator.openai_compatible || {};
  cfg.translator.openai_compatible.provider_preset = els.provider.value;
  cfg.translator.openai_compatible.base_url = els.baseUrl.value.trim();
  cfg.translator.openai_compatible.api_key = els.apiKey.value.trim();
  cfg.translator.openai_compatible.model = els.modelName.value.trim();
  cfg.translator.openai_compatible.system_prompt = els.systemPrompt.value;
  cfg.translator.openai_compatible.user_prompt_template = els.userPrompt.value;
  cfg.translator.openai_compatible.temperature = parseOptionalNumber(els.temperature.value);
  cfg.translator.openai_compatible.top_p = parseOptionalNumber(els.topP.value);
  cfg.render = cfg.render || {};
  cfg.render.text_direction = toPascalCase(els.textDirection.value || "auto");
  els.settingsJson.value = JSON.stringify(cfg, null, 2);
  state.settings = cfg;
  return cfg;
}

function toPascalCase(value) {
  const normalized = String(value || "auto").toLowerCase();
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function parseOptionalNumber(value) {
  const trimmed = String(value ?? "").trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

function applyProviderPreset() {
  const baseUrl = providerBaseUrls[els.provider.value];
  if (baseUrl) {
    els.baseUrl.value = baseUrl;
  }
  if (els.provider.value !== "Custom") {
    els.translator.value = "OpenAICompatible";
  }
  patchSettingsFromControls();
}

async function startTranslation() {
  let settings;
  try {
    settings = patchSettingsFromControls();
  } catch (err) {
    setStatus(t("jsonError"), err.message);
    addLog("error", `${t("jsonError")}: ${err.message}`);
    return;
  }

  try {
    els.startTranslation.disabled = true;
    updateProgress({ current: 0, total: state.inputPaths.length || 1, message: t("progressPreparing") });
    setStatus(t("starting"), t("backendPending"));
    const result = await invoke("startTranslation", {
      input_paths: state.inputPaths,
      settings,
      output_format: els.outputFormat.value,
    });
    renderResult(result);
    clearInputs();
    addLog("success", result.message || t("backendPending"));
  } catch (err) {
    setStatus(t("backendPending"), err.message);
    addLog("error", err.message);
  } finally {
    els.startTranslation.disabled = false;
  }
}

function renderResult(result) {
  const outputs = Array.isArray(result.outputs) ? result.outputs : [];
  state.results = outputs;
  state.selectedResults = new Set(
    outputs
      .filter((item) => item.status === "done" && item.output)
      .map((item) => item.output),
  );
  renderResults(result);
}

function renderResults(result = null) {
  if (!state.results.length) {
    els.results.className = "empty-state";
    els.results.textContent = t("resultEmpty");
    els.selectAllResults.textContent = t("selectAllResults");
    return;
  }

  els.results.className = "result-list";
  const summary = result
    ? `<div class="result-summary"><strong>${escapeHtml(result.status || "done")}</strong><p class="muted">${escapeHtml(result.message || "")}</p></div>`
    : "";
  const rows = state.results
    .map((item, index) => resultCard(item, index))
    .join("");
  els.results.innerHTML = `${summary}<div class="result-grid">${rows}</div>`;
  const doneOutputs = state.results.filter((item) => item.status === "done" && item.output);
  const allSelected =
    doneOutputs.length > 0 && doneOutputs.every((item) => state.selectedResults.has(item.output));
  els.selectAllResults.textContent = allSelected ? t("deselectAllResults") : t("selectAllResults");
}

function resultCard(item, index) {
  const output = item.output || "";
  const checked = output && state.selectedResults.has(output) ? "checked" : "";
  const canUse = item.status === "done" && output;
  const thumb = canUse && output.toLowerCase().endsWith(".png")
    ? `<div class="result-thumb"><img alt="" src="${escapeHtml(fileUrl(output))}"></div>`
    : `<div class="result-thumb muted">${escapeHtml(item.status || "-")}</div>`;
  return `
    <article class="result-item" data-status="${escapeHtml(item.status || "")}">
      <label class="result-check">
        <input type="checkbox" data-result-index="${index}" ${checked} ${canUse ? "" : "disabled"}>
        <span>${escapeHtml(item.file_name || item.status || "-")}</span>
      </label>
      ${thumb}
      <p class="result-path" title="${escapeHtml(output || item.input || "")}">${escapeHtml(output || item.input || "-")}</p>
      <p class="muted">${escapeHtml(item.message || "")}</p>
      <div class="button-row">
        <button class="ghost-button small-button" type="button" data-preview-index="${index}" ${canUse ? "" : "disabled"}>${t("preview")}</button>
      </div>
    </article>
  `;
}

function fileUrl(path) {
  return `file:///${String(path).replaceAll("\\", "/").split("/").map(encodeURIComponent).join("/")}`;
}

async function previewResult(index) {
  const item = state.results[index];
  if (!item?.output) return;
  try {
    await invoke("previewResult", { path: item.output });
  } catch (err) {
    addLog("error", err.message);
  }
}

function toggleResult(index, checked) {
  const item = state.results[index];
  if (!item?.output) return;
  if (checked) {
    state.selectedResults.add(item.output);
  } else {
    state.selectedResults.delete(item.output);
  }
  renderResults();
}

function toggleAllResults() {
  const doneOutputs = state.results.filter((item) => item.status === "done" && item.output);
  const allSelected =
    doneOutputs.length > 0 && doneOutputs.every((item) => state.selectedResults.has(item.output));
  if (allSelected) {
    doneOutputs.forEach((item) => state.selectedResults.delete(item.output));
  } else {
    doneOutputs.forEach((item) => state.selectedResults.add(item.output));
  }
  renderResults();
}

async function exportSelectedResults() {
  const outputs = [...state.selectedResults];
  if (!outputs.length) {
    setStatus(t("exportNeedSelection"), "");
    addLog("error", t("exportNeedSelection"));
    return;
  }
  if (!state.outputDir) {
    setStatus(t("exportNeedDir"), "");
    addLog("error", t("exportNeedDir"));
    return;
  }
  try {
    const data = await invoke("exportResults", {
      outputs,
      export_dir: state.outputDir,
    });
    const count = Array.isArray(data.exported) ? data.exported.length : 0;
    setStatus(t("exported"), `${count} ${t("selected")}`);
    addLog("success", `${t("exported")}: ${count}`);
  } catch (err) {
    setStatus(t("backendPending"), err.message);
    addLog("error", err.message);
  }
}

async function bootstrap() {
  applyLang();
  renderInputList();

  els.langToggle.addEventListener("click", () => {
    state.lang = state.lang === "zh" ? "en" : "zh";
    localStorage.setItem("mitWebviewLang", state.lang);
    applyLang();
  });
  els.pickImages.addEventListener("click", chooseImages);
  els.pickFolder.addEventListener("click", chooseFolder);
  els.clearInputs.addEventListener("click", clearInputs);
  els.pickOutputDir.addEventListener("click", chooseOutputDir);
  els.reloadDefaults.addEventListener("click", loadDefaults);
  els.loadConfig.addEventListener("click", loadConfig);
  els.saveConfig.addEventListener("click", saveConfig);
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      saveConfig();
    }
  });
  els.provider.addEventListener("change", applyProviderPreset);
  [
    els.translator,
    els.targetLang,
    els.textDirection,
    els.baseUrl,
    els.apiKey,
    els.modelName,
    els.temperature,
    els.topP,
    els.systemPrompt,
    els.userPrompt,
  ].forEach((node) => {
    node.addEventListener("input", () => {
      try {
        patchSettingsFromControls();
      } catch (_) {
      }
    });
    node.addEventListener("change", () => {
      try {
        patchSettingsFromControls();
      } catch (_) {
      }
    });
  });
  els.startTranslation.addEventListener("click", startTranslation);
  els.selectAllResults.addEventListener("click", toggleAllResults);
  els.exportSelected.addEventListener("click", exportSelectedResults);
  els.inputList.addEventListener("click", (event) => {
    const removeIndex = event.target?.dataset?.removeInput;
    if (removeIndex !== undefined) {
      removeInput(Number(removeIndex));
    }
  });
  els.results.addEventListener("click", (event) => {
    const previewIndex = event.target?.dataset?.previewIndex;
    if (previewIndex !== undefined) {
      previewResult(Number(previewIndex));
    }
  });
  els.results.addEventListener("change", (event) => {
    const resultIndex = event.target?.dataset?.resultIndex;
    if (resultIndex !== undefined) {
      toggleResult(Number(resultIndex), event.target.checked);
    }
  });
  els.clearLog.addEventListener("click", () => {
    els.logList.innerHTML = "";
  });

  try {
    const ready = await invoke("appReady");
    els.backendBadge.textContent = `${ready.backend} / ${ready.platform}`;
    addLog("success", `Backend bridge ready: ${ready.version}`);
  } catch (err) {
    els.backendBadge.textContent = "IPC unavailable";
    setStatus("IPC 未连接", err.message);
    addLog("error", err.message);
  }

  await loadConfig();
}

bootstrap();
