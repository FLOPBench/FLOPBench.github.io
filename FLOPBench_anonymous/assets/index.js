(function () {
  const {
    COLORS,
    basePlotlyLayout,
    emptyState,
    formatNumber,
    inlineMetric,
    metricTile,
  } = window.GFBUtils;

  const data = window.GPU_FLOWBENCH_DATA;
  if (!data) {
    console.error("gpuFLOPBench data payload is missing.");
    return;
  }

  const meta = data.meta;
  const kernelRows = data.kernelRows;
  const sourceRows = data.sourceRows;
  const explorerProgramFiles = data.explorerProgramFiles || {};
  const llmIndex = data.llmIndex || { predictionRows: [], resultShards: {} };
  const llmPredictionRows = llmIndex.predictionRows || [];
  const hasPlotly = Boolean(window.Plotly);

  const heroMetricsNode = document.getElementById("heroMetrics");
  const benchmarkSurfaceGridNode = document.getElementById("benchmarkSurfaceGrid");
  const deviceTableBody = document.getElementById("deviceTableBody");
  const downloadsGridNode = document.getElementById("downloadsGrid");
  const peakPerfListNode = document.getElementById("peakPerfList");
  const aiDenseListNode = document.getElementById("aiDenseList");
  const readingGuideMetricsNode = document.getElementById("readingGuideMetrics");
  const lastUpdatedNode = document.getElementById("lastUpdated");

  const categoryCoverageNode = document.getElementById("categoryCoverageChart");
  const rooflineNode = document.getElementById("rooflineChart");
  const rooflineSummaryNode = document.getElementById("rooflineSummary");
  const rooflineDetailSummaryNode = document.getElementById("rooflineDetailSummary");
  const rooflineDetailBody = document.getElementById("rooflineDetailBody");
  const explorerSummaryNode = document.getElementById("explorerSummary");
  const explorerKernelNameNode = document.getElementById("explorerKernelName");
  const explorerKernelMetaNode = document.getElementById("explorerKernelMeta");
  const explorerCodeSummaryNode = document.getElementById("explorerCodeSummary");
  const explorerGpuTableBody = document.getElementById("explorerGpuTableBody");
  const explorerSassCodeNode = document.getElementById("explorerSassCode");
  const explorerImixBody = document.getElementById("explorerImixBody");
  const accuracySummaryNode = document.getElementById("accuracySummary");
  const accuracyTableBody = document.getElementById("accuracyTableBody");
  const llmSummaryNode = document.getElementById("llmSummary");
  const llmMetricStripNode = document.getElementById("llmMetricStrip");
  const llmSourceOnlyMetaNode = document.getElementById("llmSourceOnlyMeta");
  const llmSassMetaNode = document.getElementById("llmSassMeta");
  const llmSourceOnlyResponseNode = document.getElementById("llmSourceOnlyResponse");
  const llmSassResponseNode = document.getElementById("llmSassResponse");
  const llmSourceOnlyPromptNode = document.getElementById("llmSourceOnlyPrompt");
  const llmSassPromptNode = document.getElementById("llmSassPrompt");

  const rooflineDevice = document.getElementById("rooflineDevice");
  const rooflineModel = document.getElementById("rooflineModel");
  const rooflineProgram = document.getElementById("rooflineProgram");
  const rooflineCategory = document.getElementById("rooflineCategory");
  const rooflineKernel = document.getElementById("rooflineKernel");
  const rooflinePrecision = document.getElementById("rooflinePrecision");
  const explorerDevice = document.getElementById("explorerDevice");
  const explorerModel = document.getElementById("explorerModel");
  const explorerCategory = document.getElementById("explorerCategory");
  const explorerSearch = document.getElementById("explorerSearch");
  const explorerProgram = document.getElementById("explorerProgram");
  const explorerKernel = document.getElementById("explorerKernel");
  const explorerArch = document.getElementById("explorerArch");
  const accuracyModel = document.getElementById("accuracyModel");
  const accuracyGpu = document.getElementById("accuracyGpu");
  const accuracyPrompt = document.getElementById("accuracyPrompt");
  const accuracyPrecision = document.getElementById("accuracyPrecision");
  const accuracyClass = document.getElementById("accuracyClass");
  const accuracySort = document.getElementById("accuracySort");
  const llmProgram = document.getElementById("llmProgram");
  const llmKernel = document.getElementById("llmKernel");
  const llmGpu = document.getElementById("llmGpu");
  const llmModel = document.getElementById("llmModel");
  const llmPrecision = document.getElementById("llmPrecision");

  function uniqueSorted(values) {
    return [...new Set(values.filter(Boolean))].sort((left, right) => left.localeCompare(right));
  }

  function shortenLabel(value, maxLength) {
    if (!value || value.length <= maxLength) {
      return value;
    }
    return `${value.slice(0, maxLength - 3)}...`;
  }

  function refillSelect(node, values, allLabel) {
    const current = node.value;
    node.innerHTML = "";

    const allOption = document.createElement("option");
    allOption.value = "all";
    allOption.textContent = allLabel;
    node.appendChild(allOption);

    values.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      node.appendChild(option);
    });

    node.value = values.includes(current) ? current : "all";
  }

  function refillChoiceSelect(node, options, emptyLabel) {
    const current = node.value;
    node.innerHTML = "";

    if (!options.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = emptyLabel;
      option.disabled = true;
      option.selected = true;
      node.appendChild(option);
      node.disabled = true;
      return "";
    }

    node.disabled = false;
    options.forEach((entry) => {
      const option = document.createElement("option");
      option.value = entry.value;
      option.textContent = entry.label;
      node.appendChild(option);
    });

    const nextValue = options.some((entry) => entry.value === current) ? current : options[0].value;
    node.value = nextValue;
    return nextValue;
  }

  const deviceArchMap = Object.fromEntries(
    (meta.roofline_specs || []).map((spec) => [
      spec.device,
      `sm_${String(spec.compute_capability).replace(/[^0-9]/g, "")}`,
    ])
  );
  const explorerProgramCache = new Map();
  const llmShardCache = new Map();
  let explorerRenderToken = 0;
  let llmRenderToken = 0;

  function buildRooflineRange(rows) {
    const aiValues = rows.map((row) => Number(row.arithmetic_intensity)).filter((value) => Number.isFinite(value) && value > 0);
    const minAI = aiValues.length ? Math.min(...aiValues) : 1e-3;
    const maxAI = aiValues.length ? Math.max(...aiValues) : 1e3;
    return {
      min: Math.pow(10, Math.floor(Math.log10(minAI)) - 0.2),
      max: Math.pow(10, Math.ceil(Math.log10(maxAI)) + 0.2),
    };
  }

  function buildLogSeries(minValue, maxValue, points) {
    const minLog = Math.log10(minValue);
    const maxLog = Math.log10(maxValue);
    return Array.from({ length: points }, function (_, index) {
      const ratio = index / (points - 1);
      return Math.pow(10, minLog + (maxLog - minLog) * ratio);
    });
  }

  function renderHeroMetrics(metrics) {
    metrics.forEach((metric) => heroMetricsNode.appendChild(metricTile(metric)));
  }

  function renderBenchmarkSurfaces() {
    const cards = [
      {
        label: "paper subset",
        title: "254 sampled kernels",
        text: `${meta.paper_subset.kernel_count} CUDA/OpenMP kernels form the paper-study subset shown on this reviewer-facing site.`,
      },
      {
        label: "profiling",
        title: "Four-GPU Roofline corpus",
        text: `${meta.paper_subset.kernel_device_rows} kernel-device rows expose profiled RAI and TFLOP/s measurements.`,
      },
      {
        label: "LLM predictions",
        title: "Source-only versus source+SASS",
        text: `${meta.paper_subset.llm_sample_count} completed LLM samples compare static source reasoning against post-compilation SASS evidence.`,
      },
      {
        label: "inspection",
        title: "Best and worst cases",
        text: `Prediction rows can be sorted by raw RAI difference, absolute error, and percent-difference error for direct inspection.`,
      },
    ];

    cards.forEach((item) => {
      const card = document.createElement("article");
      card.className = "paper-card";
      card.innerHTML = `
        <em>${item.label}</em>
        <h3>${item.title}</h3>
        <p>${item.text}</p>
      `;
      benchmarkSurfaceGridNode.appendChild(card);
    });
  }

  function renderDeviceTable(devices) {
    deviceTableBody.innerHTML = "";
    devices.forEach((device) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>
          <strong>${device.label}</strong>
          <span>${device.full_name}</span>
        </td>
        <td>${device.architecture}</td>
        <td class="mono">${device.compute_capability}</td>
        <td class="mono">${formatNumber(device.memory_bandwidth_gbps, 0)}</td>
        <td class="mono">${formatNumber(device.peak_fp16_tflops, 2)}</td>
        <td class="mono">${formatNumber(device.peak_fp32_tflops, 2)}</td>
        <td class="mono">${formatNumber(device.peak_fp64_tflops, 3)}</td>
      `;
      deviceTableBody.appendChild(tr);
    });
  }

  function renderDownloads(downloads) {
    downloads.forEach((item) => {
      const card = document.createElement("article");
      card.className = "download-card";
      card.innerHTML = `
        <span class="tag">download</span>
        <h3 class="download-title">${item.label}</h3>
        <p class="download-path">${item.path}</p>
        <div class="inline-metrics download-metrics"></div>
        <div class="download-actions">
          <a class="button secondary" href="${item.href}" download>Download artifact</a>
        </div>
      `;
      card.querySelector(".download-metrics").append(inlineMetric("size", item.size_bytes, 0));
      downloadsGridNode.appendChild(card);
    });
  }

  function renderTopList(node, title, items) {
    const card = document.createElement("details");
    card.className = "top-card top-detail";
    const preview = items[0] ? `${items[0].source} on ${items[0].device}` : "No highlighted sources yet";
    card.innerHTML = `
      <summary>
        <div class="top-summary-copy">
          <span class="tag">${title}</span>
          <h3>${title}</h3>
          <p>${preview}. Expand to inspect the highlighted source-device rows.</p>
        </div>
        <div class="top-summary-meta">
          <strong>${Math.min(items.length, 8)}</strong>
          <span>entries</span>
        </div>
      </summary>
    `;
    const list = document.createElement("div");
    list.className = "note-list";

    items.slice(0, 8).forEach((item) => {
      const row = document.createElement("div");
      row.className = "inline-metrics";
      row.style.marginTop = "10px";
      row.innerHTML = `
        <div class="inline-metric" style="flex:1 1 100%;">
          <span>${item.device} / ${item.model_type}</span>
          <strong>${item.source}</strong>
          <div class="metric-note">${item.category}</div>
        </div>
      `;
      row.append(
        inlineMetric("best perf", item.peak_performance_tflops, 4),
        inlineMetric("median RAI", item.median_arithmetic_intensity, 4),
        inlineMetric("kernels", item.kernel_count, 0)
      );
      list.appendChild(row);
    });

    card.appendChild(list);
    node.appendChild(card);
  }

  function renderReadingGuide() {
    const uniqueCategories = new Set(meta.category_profiled.map((entry) => entry.category)).size;
    readingGuideMetricsNode.append(
      inlineMetric("GPUs", meta.device_summary.length, 0),
      inlineMetric("paper kernels", meta.paper_subset.kernel_count, 0),
      inlineMetric("kernel rows", kernelRows.length, 0),
      inlineMetric("LLM rows", meta.paper_subset.llm_prediction_row_count, 0)
    );
  }



  function renderPlot(node, traces, layout, emptyMessage) {
    if (!hasPlotly) {
      emptyState(node, emptyMessage || "Interactive charts require Plotly to load.");
      return;
    }
    window.Plotly.react(node, traces, layout, { responsive: true, displayModeBar: false });
  }

  function renderCategoryCoverage(categoryProfiled) {
    const categories = [...new Set(categoryProfiled.map((entry) => entry.category))];
    const models = [...new Set(categoryProfiled.map((entry) => entry.model_type))];
    renderPlot(
      categoryCoverageNode,
      models.map((model) => ({
        type: "bar",
        orientation: "h",
        name: model.toUpperCase(),
        y: categories,
        x: categories.map((category) => {
          const entry = categoryProfiled.find((row) => row.category === category && row.model_type === model);
          return entry ? entry.profiled_sources : 0;
        }),
        marker: { color: COLORS[model] || "#90b7ff" },
      })),
      basePlotlyLayout({
        barmode: "stack",
        xaxis: { title: "profiled source binaries" },
        yaxis: { automargin: true },
        margin: { l: 180, r: 24, t: 26, b: 48 },
      })
    );
  }

  function matchesRooflineFilters(row) {
    const selectedPrecision = rooflinePrecision.value;
    const kernelLabel = row.kernel_demangled || row.kernel;
    const matchesDevice = rooflineDevice.value === "all" || row.device === rooflineDevice.value;
    const matchesModel = rooflineModel.value === "all" || row.model_type === rooflineModel.value;
    const matchesProgram = rooflineProgram.value === "all" || row.benchmark === rooflineProgram.value;
    const matchesCategory = rooflineCategory.value === "all" || row.category === rooflineCategory.value;
    const matchesKernel = rooflineKernel.value === "all" || kernelLabel === rooflineKernel.value;
    const matchesPrecision = row.dominant_precision === selectedPrecision;
    return matchesDevice && matchesModel && matchesProgram && matchesCategory && matchesKernel && matchesPrecision;
  }

  function filteredKernelRows(rows) {
    return rows.filter(matchesRooflineFilters);
  }

  function filteredRooflineContextRows(rows) {
    return rows.filter((row) => {
      const matchesDevice = rooflineDevice.value === "all" || row.device === rooflineDevice.value;
      const matchesModel = rooflineModel.value === "all" || row.model_type === rooflineModel.value;
      const matchesProgram = rooflineProgram.value === "all" || row.benchmark === rooflineProgram.value;
      const matchesCategory = rooflineCategory.value === "all" || row.category === rooflineCategory.value;
      const matchesPrecision = row.dominant_precision === rooflinePrecision.value;
      return matchesDevice && matchesModel && matchesProgram && matchesCategory && matchesPrecision;
    });
  }

  function syncRooflineFilters() {
    const baseRows = kernelRows.filter((row) => {
      const matchesDevice = rooflineDevice.value === "all" || row.device === rooflineDevice.value;
      const matchesModel = rooflineModel.value === "all" || row.model_type === rooflineModel.value;
      const matchesCategory = rooflineCategory.value === "all" || row.category === rooflineCategory.value;
      const matchesPrecision = row.dominant_precision === rooflinePrecision.value;
      return matchesDevice && matchesModel && matchesCategory && matchesPrecision;
    });

    refillSelect(rooflineProgram, uniqueSorted(baseRows.map((row) => row.benchmark)), "all programs");

    const programRows = baseRows.filter((row) => rooflineProgram.value === "all" || row.benchmark === rooflineProgram.value);
    refillSelect(rooflineKernel, uniqueSorted(programRows.map((row) => row.kernel_demangled || row.kernel)), "all kernels");
  }

  function renderRooflineDetails(rows) {
    const subset = filteredKernelRows(rows).sort((left, right) => Number(right.performance_tflops) - Number(left.performance_tflops));
    rooflineDetailBody.innerHTML = "";

    if (!subset.length) {
      rooflineDetailSummaryNode.textContent = "No exact kernel rows match the current filters.";
      return;
    }

    rooflineDetailSummaryNode.innerHTML = `
      <strong>${subset.length}</strong> exact kernel rows match the current filters.
      Narrow to a single program and kernel to inspect one row directly.
    `;

    subset.slice(0, 32).forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>
          <strong>${row.source}</strong>
          <span>${row.category}</span>
        </td>
        <td>
          <strong>${row.kernel_demangled || row.kernel}</strong>
          <span>block ${row.block_size || "n/a"} | grid ${row.grid_size || "n/a"}</span>
        </td>
        <td><span class="tag">${row.device}</span></td>
        <td><span class="tag">${row.model_type}</span></td>
        <td class="mono">${formatNumber(row.performance_tflops, 4)}</td>
        <td class="mono">${formatNumber(row.arithmetic_intensity, 4)}</td>
        <td class="mono">${formatNumber(row.float_flops, 0)}</td>
        <td class="mono">${formatNumber(row.bytes_total, 0)}</td>
        <td class="mono">${formatNumber(row.xtime_ns, 2)}</td>
      `;
      rooflineDetailBody.appendChild(tr);
    });
  }

  function renderRoofline(rows) {
    const exactRows = filteredKernelRows(rows);
    const subset = exactRows.filter((row) => Number(row.arithmetic_intensity) > 0 && Number(row.performance_tflops) > 0);
    const contextRows = filteredRooflineContextRows(rows).filter(
      (row) => Number(row.arithmetic_intensity) > 0 && Number(row.performance_tflops) > 0
    );
    renderRooflineDetails(rows);

    if (!subset.length) {
      emptyState(rooflineNode, "No floating-point kernel rows match the current filters.");
      rooflineSummaryNode.textContent = "";
      return;
    }

    const selectedPrecision = rooflinePrecision.value;
    const range = buildRooflineRange(contextRows.length ? contextRows : subset);
    const roofSpecs = meta.roofline_specs.filter((spec) => rooflineDevice.value === "all" || spec.device === rooflineDevice.value);
    const xSeries = buildLogSeries(range.min, range.max, 60);

    const roofTraces = roofSpecs.map((spec) => ({
      type: "scatter",
      mode: "lines",
      name: `${spec.device} ${selectedPrecision.toUpperCase()} roof`,
      x: xSeries,
      y: xSeries.map((ai) => Math.min(ai * spec.memory_bandwidth_gbps / 1000.0, spec[`peak_${selectedPrecision}_tflops`])),
      hovertemplate:
        `<b>${spec.label}</b><br>` +
        `precision=${selectedPrecision.toUpperCase()}<br>` +
        `bandwidth=${formatNumber(spec.memory_bandwidth_gbps, 0)} GB/s<extra></extra>`,
      line: {
        color: COLORS[spec.device] || "#90b7ff",
        width: 2,
        dash: "dash",
      },
      opacity: 0.8,
    }));

    const pointTraces = uniqueSorted(subset.map((row) => row.device)).map((device) => {
      const deviceRows = subset.filter((row) => row.device === device);
      return {
        type: "scattergl",
        mode: "markers",
        name: device,
        x: deviceRows.map((row) => row.arithmetic_intensity),
        y: deviceRows.map((row) => row.performance_tflops),
        text: deviceRows.map((row) => `${row.source}<br>${shortenLabel(row.kernel_demangled || row.kernel, 52)}`),
        customdata: deviceRows.map((row) => [row.model_type, row.dominant_precision, row.xtime_ns]),
        hovertemplate:
          "<b>%{text}</b><br>" +
          "model=%{customdata[0]}<br>" +
          "dominant precision=%{customdata[1]}<br>" +
          "RAI=%{x:.4f}<br>" +
          "performance=%{y:.4f} TFLOP/s<br>" +
          "time=%{customdata[2]:.2f} ns<extra></extra>",
        marker: {
          size: 8,
          opacity: 0.72,
          color: COLORS[device] || "#90b7ff",
        },
      };
    });

    renderPlot(
      rooflineNode,
      roofTraces.concat(pointTraces),
      basePlotlyLayout({
        xaxis: { title: "RAI (FLOPs / byte)", type: "log" },
        yaxis: { title: "performance (TFLOP/s)", type: "log" },
        margin: { l: 64, r: 28, t: 30, b: 58 },
      })
    );

    const perfValues = subset.map((row) => Number(row.performance_tflops)).sort((left, right) => left - right);
    const aiValues = subset.map((row) => Number(row.arithmetic_intensity)).sort((left, right) => left - right);
    rooflineSummaryNode.innerHTML = `
      <strong>${subset.length}</strong> floating-point kernel rows in view.
      Median RAI <strong>${formatNumber(aiValues[Math.floor(aiValues.length / 2)], 4)}</strong>,
      median performance <strong>${formatNumber(perfValues[Math.floor(perfValues.length / 2)], 4)} TFLOP/s</strong>.
      Dashed lines show the ${selectedPrecision.toUpperCase()} theoretical roofline at default device clocks.
    `;
  }

  function filteredExplorerKernelRows(rows) {
    const search = explorerSearch.value.trim().toLowerCase();
    return rows.filter((row) => {
      const matchesDevice = explorerDevice.value === "all" || row.device === explorerDevice.value;
      const matchesModel = explorerModel.value === "all" || row.model_type === explorerModel.value;
      const matchesCategory = explorerCategory.value === "all" || row.category === explorerCategory.value;
      const matchesSearch =
        !search ||
        row.source.toLowerCase().includes(search) ||
        row.benchmark.toLowerCase().includes(search) ||
        String(row.kernel_demangled || row.kernel || "").toLowerCase().includes(search) ||
        String(row.kernel_symbol || "").toLowerCase().includes(search);
      return matchesDevice && matchesModel && matchesCategory && matchesSearch;
    });
  }

  function buildExplorerPairs(rows) {
    const pairs = new Map();
    rows.forEach((row) => {
      const key = `${row.source}::${row.kernel_symbol}`;
      if (!pairs.has(key)) {
        pairs.set(key, {
          program: row.source,
          benchmark: row.benchmark,
          category: row.category,
          model_type: row.model_type,
          kernel_symbol: row.kernel_symbol,
          kernel_demangled: row.kernel_demangled || row.kernel || row.kernel_symbol,
          exe_args: row.exe_args,
        });
      }
    });
    return [...pairs.values()].sort((left, right) => {
      const programOrder = left.program.localeCompare(right.program);
      if (programOrder !== 0) {
        return programOrder;
      }
      return left.kernel_demangled.localeCompare(right.kernel_demangled);
    });
  }

  async function loadExplorerProgram(program) {
    if (!program) {
      throw new Error("No program was selected for the Source Explorer.");
    }
    if (explorerProgramCache.has(program)) {
      return explorerProgramCache.get(program);
    }

    const relativePath = explorerProgramFiles[program];
    if (!relativePath) {
      throw new Error(`No explorer payload is registered for program ${program}.`);
    }

    const promise = fetch(`./${relativePath}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to fetch ${relativePath} (${response.status}).`);
        }
        return response.json();
      })
      .catch((error) => {
        explorerProgramCache.delete(program);
        throw error;
      });

    explorerProgramCache.set(program, promise);
    return promise;
  }

  function renderExplorerGpuTable(rows, selectedArch) {
    explorerGpuTableBody.innerHTML = "";

    if (!rows.length) {
      const tr = document.createElement("tr");
      tr.innerHTML = '<td class="muted" colspan="8">No GPU measurements are available for the selected kernel.</td>';
      explorerGpuTableBody.appendChild(tr);
      return;
    }

    rows.forEach((row) => {
      const arch = deviceArchMap[row.device] || "n/a";
      const tr = document.createElement("tr");
      tr.className = `table-row-selectable${arch === selectedArch ? " table-row-selected" : ""}`;
      tr.innerHTML = `
        <td><span class="tag">${row.device}</span></td>
        <td class="mono">${arch}</td>
        <td class="mono">${formatNumber(row.float_flops, 0)}</td>
        <td class="mono">${formatNumber(row.bytes_total, 0)}</td>
        <td class="mono">${formatNumber(row.performance_tflops, 4)}</td>
        <td class="mono">${formatNumber(row.xtime_ns, 2)}</td>
        <td class="mono">${row.grid_size || "n/a"}</td>
        <td class="mono">${row.block_size || "n/a"}</td>
        <td class="mono">${row.exe_args || "n/a"}</td>
      `;
      tr.addEventListener("click", function () {
        if (arch && arch !== "n/a") {
          explorerArch.value = arch;
          renderExplorer(kernelRows);
        }
      });
      explorerGpuTableBody.appendChild(tr);
    });
  }

  function renderExplorerSass(sections, kernelSymbol) {
    if (!sections) {
      emptyState(explorerSassCodeNode, "No SASS disassembly is available for the selected architecture.");
      return;
    }

    if (typeof sections === "string") {
      explorerSassCodeNode.textContent = sections;
      return;
    }

    const rendered = Object.entries(sections).map(([sectionName, code]) => {
      const header = sectionName === kernelSymbol ? `${sectionName} (kernel entry)` : sectionName;
      return `// [${header}]\n${code}`;
    });
    explorerSassCodeNode.textContent = rendered.join("\n\n");
  }

  function renderExplorerImix(imix, emptyMessage) {
    explorerImixBody.innerHTML = "";

    if (!imix || !Object.keys(imix).length) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td class="muted" colspan="2">${emptyMessage || "No IMIX is available for the selected architecture."}</td>`;
      explorerImixBody.appendChild(tr);
      return;
    }

    Object.entries(imix)
      .sort((left, right) => {
        if (right[1] !== left[1]) {
          return right[1] - left[1];
        }
        return left[0].localeCompare(right[0]);
      })
      .forEach(([instruction, count]) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td class="mono">${instruction}</td>
          <td class="mono">${formatNumber(count, 0)}</td>
        `;
        explorerImixBody.appendChild(tr);
      });
  }

  async function renderExplorerCodePane(selectedPair, selectedRows, selectedArch, renderToken) {
    explorerCodeSummaryNode.textContent = "Loading SASS and IMIX for the selected program.";
    emptyState(explorerSassCodeNode, "Loading kernel SASS...");
    renderExplorerImix(null, "Loading kernel IMIX...");

    try {
      const programPayload = await loadExplorerProgram(selectedPair.program);
      if (renderToken !== explorerRenderToken) {
        return;
      }

      const rawKernel = programPayload?.kernels?.[selectedPair.kernel_symbol];
      if (!rawKernel) {
        explorerCodeSummaryNode.textContent = "The selected kernel was not found in the generated explorer payload.";
        emptyState(explorerSassCodeNode, "The selected kernel was not found in the per-program explorer data.");
        renderExplorerImix(null);
        return;
      }

      const availableArchs = uniqueSorted(
        [
          ...selectedRows.map((row) => deviceArchMap[row.device]).filter(Boolean),
          ...Object.keys(rawKernel.sass_code || {}),
          ...Object.keys(rawKernel.imix || {}),
        ]
      );
      const activeArch = refillChoiceSelect(
        explorerArch,
        availableArchs.map((arch) => ({ value: arch, label: arch })),
        "no architecture"
      );
      const sassSections = rawKernel.sass_code ? rawKernel.sass_code[activeArch] : null;
      const imix = rawKernel.imix ? rawKernel.imix[activeArch] : null;
      const sectionCount =
        sassSections && typeof sassSections === "object" ? Object.keys(sassSections).length : sassSections ? 1 : 0;

      explorerCodeSummaryNode.textContent = `${activeArch || "n/a"} selected. ${sectionCount} SASS section${
        sectionCount === 1 ? "" : "s"
      } and the matching IMIX are shown for this kernel.`;
      renderExplorerSass(sassSections, selectedPair.kernel_symbol);
      renderExplorerImix(imix);
      renderExplorerGpuTable(selectedRows, activeArch);
    } catch (error) {
      if (renderToken !== explorerRenderToken) {
        return;
      }
      console.error(error);
      explorerCodeSummaryNode.textContent = "The per-program explorer payload could not be loaded.";
      emptyState(explorerSassCodeNode, "Could not load kernel SASS for the selected program.");
      renderExplorerImix(null, "Could not load kernel IMIX for the selected program.");
    }
  }

  function renderExplorer(rows) {
    const filteredRows = filteredExplorerKernelRows(rows);
    const pairs = buildExplorerPairs(filteredRows);
    const programNames = uniqueSorted(pairs.map((pair) => pair.program));
    const selectedProgram = refillChoiceSelect(
      explorerProgram,
      programNames.map((program) => ({ value: program, label: program })),
      "no matching programs"
    );
    const selectedProgramPairs = pairs.filter((pair) => pair.program === selectedProgram);
    const selectedKernelSymbol = refillChoiceSelect(
      explorerKernel,
      selectedProgramPairs.map((pair) => ({
        value: pair.kernel_symbol,
        label: pair.kernel_demangled,
      })),
      "no matching kernels"
    );
    const selectedPair = selectedProgramPairs.find((pair) => pair.kernel_symbol === selectedKernelSymbol);
    const selectedRows = filteredRows
      .filter((row) => row.source === selectedProgram && row.kernel_symbol === selectedKernelSymbol)
      .sort((left, right) => left.device.localeCompare(right.device));

    const rowArchs = uniqueSorted(selectedRows.map((row) => deviceArchMap[row.device]).filter(Boolean));
    refillChoiceSelect(
      explorerArch,
      rowArchs.map((arch) => ({ value: arch, label: arch })),
      "no architecture"
    );

    explorerSummaryNode.innerHTML = `<strong>${pairs.length}</strong> program-kernel pairs across <strong>${
      programNames.length
    }</strong> programs match the current filters.`;

    if (!selectedPair || !selectedRows.length) {
      explorerKernelNameNode.textContent = "No kernel matches the current filters.";
      explorerKernelMetaNode.textContent = "Adjust the Source Explorer filters to select a profiled program and kernel.";
      explorerCodeSummaryNode.textContent = "Waiting for a matching kernel selection.";
      renderExplorerGpuTable([], "");
      emptyState(explorerSassCodeNode, "Select a profiled kernel to inspect SASS.");
      renderExplorerImix(null, "Select a profiled kernel to inspect IMIX.");
      return;
    }

    explorerKernelNameNode.textContent = selectedPair.kernel_demangled;
    explorerKernelMetaNode.textContent = `${selectedPair.program} | ${selectedPair.model_type.toUpperCase()} | ${
      selectedPair.category
    } | benchmark ${selectedPair.benchmark} | symbol ${selectedPair.kernel_symbol}`;
    renderExplorerGpuTable(selectedRows, explorerArch.value);
    renderExplorerCodePane(selectedPair, selectedRows, explorerArch.value, ++explorerRenderToken);
  }

  function formatPercent(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "undefined";
    }
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return "undefined";
    }
    return `${numeric.toFixed(Math.abs(numeric) >= 100 ? 1 : 2)}%`;
  }

  function kernelKey(row) {
    return `${row.program_name}::${row.kernel_mangled_name}`;
  }

  function filteredAccuracyRows() {
    return llmPredictionRows.filter((row) => {
      const matchesModel = accuracyModel.value === "all" || row.model_name === accuracyModel.value;
      const matchesGpu = accuracyGpu.value === "all" || row.gpu === accuracyGpu.value;
      const matchesPrompt = accuracyPrompt.value === "all" || row.prompt_type === accuracyPrompt.value;
      const matchesPrecision = accuracyPrecision.value === "all" || row.precision === accuracyPrecision.value;
      const matchesClass =
        accuracyClass.value === "all" ||
        row.expected_bound === accuracyClass.value ||
        (accuracyClass.value === "nonzero" && row.expected_bound !== "zero");
      return matchesModel && matchesGpu && matchesPrompt && matchesPrecision && matchesClass;
    });
  }

  function accuracySortValue(row, key) {
    if (key === "largest_under") {
      return Number(row.ai_raw_diff);
    }
    if (key === "largest_over") {
      return -Number(row.ai_raw_diff);
    }
    if (key === "worst_abs") {
      return -Number(row.ai_abs_error);
    }
    if (key === "best_ape") {
      return Number.isFinite(Number(row.ai_abs_percent_error)) ? Number(row.ai_abs_percent_error) : Number.POSITIVE_INFINITY;
    }
    return Number.isFinite(Number(row.ai_abs_percent_error)) ? -Number(row.ai_abs_percent_error) : Number.POSITIVE_INFINITY;
  }

  function sortAccuracyRows(rows) {
    const sortKey = accuracySort.value;
    return rows.slice().sort((left, right) => {
      const leftValue = accuracySortValue(left, sortKey);
      const rightValue = accuracySortValue(right, sortKey);
      if (leftValue !== rightValue) {
        return leftValue - rightValue;
      }
      return `${left.program_name}${left.kernel_demangled_name}`.localeCompare(`${right.program_name}${right.kernel_demangled_name}`);
    });
  }

  function selectLlmRow(row) {
    llmProgram.value = row.program_name;
    syncLlmFilters();
    llmKernel.value = row.kernel_mangled_name;
    syncLlmFilters();
    llmGpu.value = row.gpu;
    llmModel.value = row.model_name;
    llmPrecision.value = row.precision;
    renderLlmExplorer();
    document.getElementById("llm-explorer").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderAccuracyBrowser() {
    if (!llmPredictionRows.length) {
      accuracySummaryNode.textContent = "No LLM prediction rows are available. Run scripts/export_paper_llm_data.py before building the site.";
      accuracyTableBody.innerHTML = "";
      return;
    }

    const rows = sortAccuracyRows(filteredAccuracyRows());
    accuracyTableBody.innerHTML = "";
    accuracySummaryNode.innerHTML = `
      <strong>${rows.length}</strong> prediction rows in view across <strong>${llmIndex.kernel_count || 0}</strong> sampled kernels.
      Click a row to open its source-only versus source+SASS response comparison.
    `;

    rows.slice(0, 80).forEach((row) => {
      const tr = document.createElement("tr");
      tr.className = "table-row-selectable";
      tr.innerHTML = `
        <td>
          <strong>${row.program_name}</strong>
          <span>${shortenLabel(row.kernel_demangled_name || row.kernel_mangled_name, 72)}</span>
        </td>
        <td><span class="tag">${row.gpu}</span></td>
        <td>${row.model_name}</td>
        <td>${row.prompt_type}</td>
        <td class="mono">${row.precision.toUpperCase()}</td>
        <td class="mono">${formatNumber(row.expected_ai, 5)}</td>
        <td class="mono">${formatNumber(row.predicted_ai, 5)}</td>
        <td class="mono">${formatNumber(row.ai_raw_diff, 5)}</td>
        <td class="mono">${formatNumber(row.ai_abs_error, 5)}</td>
        <td class="mono">${formatPercent(row.ai_percent_diff)}</td>
      `;
      tr.addEventListener("click", function () {
        selectLlmRow(row);
      });
      accuracyTableBody.appendChild(tr);
    });
  }

  function syncLlmFilters() {
    const programs = uniqueSorted(llmPredictionRows.map((row) => row.program_name));
    const selectedProgram = refillChoiceSelect(
      llmProgram,
      programs.map((program) => ({ value: program, label: program })),
      "no programs"
    );

    const programRows = llmPredictionRows.filter((row) => row.program_name === selectedProgram);
    const kernels = [];
    const seenKernels = new Set();
    programRows.forEach((row) => {
      if (seenKernels.has(row.kernel_mangled_name)) {
        return;
      }
      seenKernels.add(row.kernel_mangled_name);
      kernels.push({
        value: row.kernel_mangled_name,
        label: row.kernel_demangled_name || row.kernel_mangled_name,
      });
    });
    const selectedKernel = refillChoiceSelect(llmKernel, kernels, "no kernels");
    const kernelRowsForSelection = programRows.filter((row) => row.kernel_mangled_name === selectedKernel);

    refillChoiceSelect(
      llmGpu,
      uniqueSorted(kernelRowsForSelection.map((row) => row.gpu)).map((gpu) => ({ value: gpu, label: gpu })),
      "no GPUs"
    );
    refillChoiceSelect(
      llmModel,
      uniqueSorted(kernelRowsForSelection.map((row) => row.model_name)).map((model) => ({ value: model, label: model })),
      "no models"
    );
  }

  async function loadLlmShard(program) {
    const shardPath = llmIndex.resultShards ? llmIndex.resultShards[program] : null;
    if (!shardPath) {
      throw new Error(`No LLM result shard is registered for ${program}.`);
    }
    if (llmShardCache.has(shardPath)) {
      return llmShardCache.get(shardPath);
    }
    const promise = fetch(`./${shardPath}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to fetch ${shardPath} (${response.status}).`);
        }
        return response.json();
      })
      .catch((error) => {
        llmShardCache.delete(shardPath);
        throw error;
      });
    llmShardCache.set(shardPath, promise);
    return promise;
  }

  function renderJsonObject(node, value) {
    node.innerHTML = "";
    if (!value || (typeof value === "object" && !Object.keys(value).length)) {
      emptyState(node, "No structured response is available for this selection.");
      return;
    }
    const pre = document.createElement("pre");
    pre.className = "json-block mono";
    pre.textContent = JSON.stringify(value, null, 2);
    node.appendChild(pre);
  }

  function selectedLlmPredictionRow(promptType) {
    return llmPredictionRows.find(
      (row) =>
        row.program_name === llmProgram.value &&
        row.kernel_mangled_name === llmKernel.value &&
        row.gpu === llmGpu.value &&
        row.model_name === llmModel.value &&
        row.precision === llmPrecision.value &&
        row.prompt_type === promptType
    );
  }

  function renderLlmMetrics(sourceOnlyRow, sassRow) {
    llmMetricStripNode.innerHTML = "";
    const rows = [sourceOnlyRow, sassRow].filter(Boolean);
    if (!rows.length) {
      return;
    }
    rows.forEach((row) => {
      llmMetricStripNode.append(
        inlineMetric(`${row.prompt_type} expected`, row.expected_ai, 5),
        inlineMetric(`${row.prompt_type} predicted`, row.predicted_ai, 5),
        inlineMetric(`${row.prompt_type} diff`, row.ai_raw_diff, 5),
        inlineMetric(`${row.prompt_type} pct`, row.ai_percent_diff, 2)
      );
    });
  }

  function renderLlmPanel(record, row, prompt, metaNode, responseNode, promptNode) {
    if (!record || !row) {
      metaNode.textContent = "No completed record is available for this selection.";
      promptNode.textContent = "";
      renderJsonObject(responseNode, null);
      return;
    }
    metaNode.textContent = `${row.model_name} on ${row.gpu}, ${row.precision.toUpperCase()}: raw diff ${formatNumber(
      row.ai_raw_diff,
      5
    )}, abs error ${formatNumber(row.ai_abs_error, 5)}, percent diff ${formatPercent(row.ai_percent_diff)}.`;
    renderJsonObject(responseNode, record.prediction || record.raw_response);
    const systemPrompt = prompt?.system_prompt || "";
    const humanPrompt = prompt?.human_prompt || "";
    promptNode.textContent = `--- System Prompt ---\n${systemPrompt}\n\n--- Human Prompt ---\n${humanPrompt}`;
  }

  async function renderLlmExplorer() {
    if (!llmPredictionRows.length) {
      llmSummaryNode.textContent = "No LLM response data is available.";
      return;
    }

    syncLlmFilters();
    const token = ++llmRenderToken;
    const sourceOnlyRow = selectedLlmPredictionRow("Source-Only");
    const sassRow = selectedLlmPredictionRow("Source+SASS");
    const representativeRow = sourceOnlyRow || sassRow;
    if (!representativeRow) {
      llmSummaryNode.textContent = "No source-only or source+SASS row matches the current selection.";
      renderLlmMetrics(null, null);
      renderLlmPanel(null, null, llmSourceOnlyMetaNode, llmSourceOnlyResponseNode, llmSourceOnlyPromptNode);
      renderLlmPanel(null, null, llmSassMetaNode, llmSassResponseNode, llmSassPromptNode);
      return;
    }

    llmSummaryNode.textContent = "Loading prompt and response shard for the selected program.";
    renderLlmMetrics(sourceOnlyRow, sassRow);
    try {
      const shard = await loadLlmShard(representativeRow.program_name);
      if (token !== llmRenderToken) {
        return;
      }
      const records = shard.records || [];
      const prompts = shard.prompts || {};
      const findRecord = (row) =>
        row
          ? records.find(
              (record) =>
                record.thread_id === row.thread_id ||
                (record.kernel_mangled_name === row.kernel_mangled_name &&
                  record.gpu === row.gpu &&
                  record.model_name === row.model_name &&
                  record.prompt_type === row.prompt_type)
            )
          : null;
      const sourceOnlyRecord = findRecord(sourceOnlyRow);
      const sassRecord = findRecord(sassRow);
      renderLlmPanel(
        sourceOnlyRecord,
        sourceOnlyRow,
        sourceOnlyRecord ? prompts[sourceOnlyRecord.prompt_key] : null,
        llmSourceOnlyMetaNode,
        llmSourceOnlyResponseNode,
        llmSourceOnlyPromptNode
      );
      renderLlmPanel(
        sassRecord,
        sassRow,
        sassRecord ? prompts[sassRecord.prompt_key] : null,
        llmSassMetaNode,
        llmSassResponseNode,
        llmSassPromptNode
      );
      llmSummaryNode.innerHTML = `
        Showing <strong>${representativeRow.program_name}</strong> /
        <strong>${shortenLabel(representativeRow.kernel_demangled_name || representativeRow.kernel_mangled_name, 80)}</strong>
        for <strong>${representativeRow.gpu}</strong>, <strong>${representativeRow.model_name}</strong>,
        <strong>${representativeRow.precision.toUpperCase()}</strong>.
      `;
    } catch (error) {
      console.error(error);
      llmSummaryNode.textContent = "Could not load the LLM response shard for this selection.";
    }
  }

  function init() {
    renderHeroMetrics(meta.hero.headline_metrics);
    renderBenchmarkSurfaces();
    renderDeviceTable(meta.device_summary);
    renderDownloads(meta.downloads);
    renderTopList(peakPerfListNode, "Performance leaders", meta.top_lists.performance_sources);
    renderTopList(aiDenseListNode, "RAI-dense leaders", meta.top_lists.ai_dense_sources);
    renderReadingGuide();
    renderCategoryCoverage(meta.category_profiled);

    refillSelect(rooflineDevice, uniqueSorted(kernelRows.map((row) => row.device)), "all devices");
    refillSelect(rooflineModel, uniqueSorted(kernelRows.map((row) => row.model_type)), "all models");
    refillSelect(rooflineCategory, uniqueSorted(kernelRows.map((row) => row.category)), "all categories");
    syncRooflineFilters();

    refillSelect(explorerDevice, uniqueSorted(kernelRows.map((row) => row.device)), "all devices");
    refillSelect(explorerModel, uniqueSorted(kernelRows.map((row) => row.model_type)), "all models");
    refillSelect(explorerCategory, uniqueSorted(kernelRows.map((row) => row.category)), "all categories");
    refillSelect(accuracyModel, uniqueSorted(llmPredictionRows.map((row) => row.model_name)), "all models");
    refillSelect(accuracyGpu, uniqueSorted(llmPredictionRows.map((row) => row.gpu)), "all GPUs");
    refillSelect(accuracyPrompt, uniqueSorted(llmPredictionRows.map((row) => row.prompt_type)), "all prompts");
    refillSelect(accuracyPrecision, uniqueSorted(llmPredictionRows.map((row) => row.precision)), "all precisions");
    syncLlmFilters();

    [rooflineDevice, rooflineModel, rooflineProgram, rooflineCategory, rooflineKernel, rooflinePrecision].forEach((node) => {
      node.addEventListener("change", function () {
        syncRooflineFilters();
        renderRoofline(kernelRows);
      });
    });

    [explorerDevice, explorerModel, explorerCategory, explorerProgram, explorerKernel, explorerArch].forEach((node) => {
      node.addEventListener("change", function () {
        renderExplorer(kernelRows);
      });
    });
    explorerSearch.addEventListener("input", function () {
      renderExplorer(kernelRows);
    });
    [accuracyModel, accuracyGpu, accuracyPrompt, accuracyPrecision, accuracyClass, accuracySort].forEach((node) => {
      node.addEventListener("change", renderAccuracyBrowser);
    });
    [llmProgram, llmKernel, llmGpu, llmModel, llmPrecision].forEach((node) => {
      node.addEventListener("change", renderLlmExplorer);
    });

    renderRoofline(kernelRows);
    renderAccuracyBrowser();
    renderLlmExplorer();
    renderExplorer(kernelRows);
    lastUpdatedNode.textContent = new Date(meta.audit.generated_at).toLocaleString();
  }

  init();
})();
