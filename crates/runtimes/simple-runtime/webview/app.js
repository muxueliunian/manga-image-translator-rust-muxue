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
    dropTitle: "从这里开始：选择漫画页、单张图或整本文件夹",
    dropHint: "支持多选图片或选择整本文件夹，翻译完成后会按所选格式写入输出目录。",
    outputDir: "输出目录",
    outputFormat: "输出格式",
    configKicker: "模型",
    configTitle: "翻译配置",
    reloadDefaults: "默认值",
    translator: "翻译器",
    targetLang: "目标语言",
    provider: "模型供应商",
    modelName: "模型名称",
    advancedJson: "高级 JSON 配置",
    readyTitle: "准备就绪",
    readyText: "选择输入与输出目录后即可开始。",
    start: "开始翻译",
    resultKicker: "结果",
    resultTitle: "任务输出",
    resultEmpty: "暂无结果。完成翻译后会显示输出图片路径。",
    logKicker: "日志",
    logTitle: "运行记录",
    clearLog: "清空",
    noInput: "未选择输入",
    selected: "已选择",
    folderSelected: "已选择文件夹",
    outputSelected: "输出目录已设置",
    defaultsLoaded: "默认配置已加载",
    starting: "已发送任务",
    backendPending: "正在执行翻译任务",
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
    dropTitle: "Start here: choose manga pages, one image, or a whole folder",
    dropHint: "Select multiple images or an entire folder. Finished pages are written to the output directory.",
    outputDir: "Output Directory",
    outputFormat: "Output Format",
    configKicker: "Models",
    configTitle: "Translation Config",
    reloadDefaults: "Defaults",
    translator: "Translator",
    targetLang: "Target Language",
    provider: "Provider",
    modelName: "Model Name",
    advancedJson: "Advanced JSON Config",
    readyTitle: "Ready",
    readyText: "Choose input and output directory to begin.",
    start: "Start Translating",
    resultKicker: "Results",
    resultTitle: "Task Output",
    resultEmpty: "No results yet. Output image paths will appear after translation.",
    logKicker: "Logs",
    logTitle: "Run Log",
    clearLog: "Clear",
    noInput: "No input selected",
    selected: "selected",
    folderSelected: "Folder selected",
    outputSelected: "Output directory set",
    defaultsLoaded: "Default settings loaded",
    starting: "Job sent",
    backendPending: "Translation is running",
    jsonError: "Invalid JSON settings",
  },
};

const state = {
  lang: localStorage.getItem("mitWebviewLang") || "zh",
  inputPaths: [],
  outputDir: "",
  settings: null,
  requestId: 0,
  pending: new Map(),
};

const els = {
  langToggle: document.getElementById("langToggle"),
  backendBadge: document.getElementById("backendBadge"),
  pickImages: document.getElementById("pickImages"),
  pickFolder: document.getElementById("pickFolder"),
  pickOutputDir: document.getElementById("pickOutputDir"),
  outputDir: document.getElementById("outputDir"),
  outputFormat: document.getElementById("outputFormat"),
  inputList: document.getElementById("inputList"),
  translator: document.getElementById("translator"),
  targetLang: document.getElementById("targetLang"),
  provider: document.getElementById("provider"),
  modelName: document.getElementById("modelName"),
  settingsJson: document.getElementById("settingsJson"),
  reloadDefaults: document.getElementById("reloadDefaults"),
  startTranslation: document.getElementById("startTranslation"),
  statusTitle: document.getElementById("statusTitle"),
  statusText: document.getElementById("statusText"),
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

function renderInputList() {
  els.inputList.innerHTML = "";
  if (!state.inputPaths.length) {
    const empty = document.createElement("div");
    empty.className = "path-item";
    empty.textContent = t("noInput");
    els.inputList.append(empty);
    return;
  }

  state.inputPaths.forEach((path) => {
    const item = document.createElement("div");
    item.className = "path-item";
    const name = path.split(/[\\/]/).filter(Boolean).pop() || path;
    item.innerHTML = `<strong title="${escapeHtml(path)}">${escapeHtml(name)}</strong><span>${escapeHtml(path)}</span>`;
    els.inputList.append(item);
  });
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
    state.settings = defaults;
    els.settingsJson.value = JSON.stringify(defaults, null, 2);
    syncControlsFromSettings();
    addLog("success", t("defaultsLoaded"));
  } catch (err) {
    addLog("error", err.message);
  }
}

function syncControlsFromSettings() {
  const cfg = state.settings || {};
  const translation = cfg.translator?.target?.translator ? cfg.translator.target : null;
  els.translator.value = translation?.translator || "Sugoi";
  els.targetLang.value = translation?.target || "en";
  els.provider.value = cfg.translator?.openai_compatible?.provider_preset || "Custom";
  els.modelName.value = cfg.translator?.openai_compatible?.model || "";
}

function patchSettingsFromControls() {
  const cfg = JSON.parse(els.settingsJson.value || "{}");
  cfg.translator = cfg.translator || {};
  cfg.translator.target = cfg.translator.target || {};
  cfg.translator.target.translator = els.translator.value;
  cfg.translator.target.target = els.targetLang.value;
  cfg.translator.openai_compatible = cfg.translator.openai_compatible || {};
  cfg.translator.openai_compatible.provider_preset = els.provider.value;
  cfg.translator.openai_compatible.model = els.modelName.value.trim();
  els.settingsJson.value = JSON.stringify(cfg, null, 2);
  state.settings = cfg;
  return cfg;
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
    setStatus(t("starting"), t("backendPending"));
    const result = await invoke("startTranslation", {
      input_paths: state.inputPaths,
      output_dir: state.outputDir || null,
      settings,
      output_format: els.outputFormat.value,
    });
    renderResult(result);
    addLog("success", result.message || t("backendPending"));
  } catch (err) {
    setStatus(t("backendPending"), err.message);
    addLog("error", err.message);
  } finally {
    els.startTranslation.disabled = false;
  }
}

function renderResult(result) {
  els.results.className = "";
  const outputs = Array.isArray(result.outputs) ? result.outputs : [];
  const rows = outputs.map((item) => {
    const output = item.output || "-";
    return `
      <div class="result-item" data-status="${escapeHtml(item.status || "")}">
        <strong>${escapeHtml(item.status || "")}</strong>
        <p>${escapeHtml(item.input || "")}</p>
        <p class="muted">${escapeHtml(output)}</p>
        <p class="muted">${escapeHtml(item.message || "")}</p>
      </div>
    `;
  }).join("");
  els.results.innerHTML = `
    <div class="result-summary">
      <strong>${escapeHtml(result.status || "done")}</strong>
      <p class="muted">${escapeHtml(result.message || "")}</p>
    </div>
    <div class="result-list">${rows}</div>
  `;
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
  els.pickOutputDir.addEventListener("click", chooseOutputDir);
  els.reloadDefaults.addEventListener("click", loadDefaults);
  els.startTranslation.addEventListener("click", startTranslation);
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

  await loadDefaults();
}

bootstrap();
