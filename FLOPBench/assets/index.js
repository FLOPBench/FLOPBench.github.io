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
  const hasPlotly = Boolean(window.Plotly);

  const heroMetricsNode = document.getElementById("heroMetrics");
  const benchmarkSurfaceGridNode = document.getElementById("benchmarkSurfaceGrid");
  const deviceCardsNode = document.getElementById("deviceCards");
  const downloadsGridNode = document.getElementById("downloadsGrid");
  const peakPerfListNode = document.getElementById("peakPerfList");
  const aiDenseListNode = document.getElementById("aiDenseList");
  const readingGuideMetricsNode = document.getElementById("readingGuideMetrics");
  const citationMetaNode = document.getElementById("citationMeta");
  const citationBibtexNode = document.getElementById("citationBibtex");
  const teamGridNode = document.getElementById("teamGrid");
  const lastUpdatedNode = document.getElementById("lastUpdated");

  const modelCoverageNode = document.getElementById("modelCoverageChart");
  const categoryCoverageNode = document.getElementById("categoryCoverageChart");
  const devicePerfNode = document.getElementById("devicePerfChart");
  const rooflineNode = document.getElementById("rooflineChart");
  const rooflineSummaryNode = document.getElementById("rooflineSummary");
  const rooflineReferenceSummaryNode = document.getElementById("rooflineReferenceSummary");
  const rooflineReferencePreviewNode = document.getElementById("rooflineReferencePreview");
  const rooflineReferenceCountNode = document.getElementById("rooflineReferenceCount");
  const devicePerfPanelNode = document.getElementById("devicePerfPanel");
  const rooflineReferencePanelNode = document.getElementById("rooflineReferencePanel");
  const rooflineReferenceBodyNode = document.getElementById("rooflineReferenceBody");
  const rooflineAtlasPanelNode = document.getElementById("rooflineAtlasPanel");
  const rooflineSpecGridNode = document.getElementById("rooflineSpecGrid");
  const rooflineDetailSummaryNode = document.getElementById("rooflineDetailSummary");
  const rooflineDetailBody = document.getElementById("rooflineDetailBody");
  const explorerSummaryNode = document.getElementById("explorerSummary");
  const explorerKernelNameNode = document.getElementById("explorerKernelName");
  const explorerKernelMetaNode = document.getElementById("explorerKernelMeta");
  const explorerCodeSummaryNode = document.getElementById("explorerCodeSummary");
  const explorerGpuTableBody = document.getElementById("explorerGpuTableBody");
  const explorerSassCodeNode = document.getElementById("explorerSassCode");
  const explorerImixBody = document.getElementById("explorerImixBody");

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
  let explorerDatasetPromise = null;
  let explorerRenderToken = 0;

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
        label: "inventory",
        title: "Benchmark inventory",
        text: `${meta.inventory.totals.benchmarks_yaml} benchmark entries define the source footprint visible in gpuFLOPBench.`,
      },
      {
        label: "profiling",
        title: "Kernel performance corpus",
        text: `${meta.inventory.totals.profiled_sources} profiled source binaries expand into ${kernelRows.length} exact kernel-device rows.`,
      },
      {
        label: "roofline",
        title: "Measured floating-point rooflines",
        text: `Kernel performance is recomputed from floating-point work over execution time and plotted against arithmetic intensity.`,
      },
      {
        label: "exploration",
        title: "Source and kernel drill-down",
        text: `The site keeps category coverage, exact kernel rows, and source-level best-observed performance in one place.`,
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

  function renderDeviceCards(devices) {
    devices.forEach((device) => {
      const card = document.createElement("article");
      card.className = "device-card";
      card.innerHTML = `
        <span class="tag">${device.device}</span>
        <h3>${device.label}</h3>
        <p>${device.architecture}, compute capability ${device.compute_capability}.</p>
      `;
      const metrics = document.createElement("div");
      metrics.className = "inline-metrics";
      metrics.append(
        inlineMetric("sources", device.sources, 0),
        inlineMetric("kernels", device.kernels, 0),
        inlineMetric("bandwidth (GB/s)", device.memory_bandwidth_gbps, 0),
        inlineMetric("peak fp32", device.peak_fp32_tflops, 2)
      );
      card.appendChild(metrics);
      deviceCardsNode.appendChild(card);
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
        inlineMetric("median AI", item.median_arithmetic_intensity, 4),
        inlineMetric("kernels", item.kernel_count, 0)
      );
      list.appendChild(row);
    });

    card.appendChild(list);
    node.appendChild(card);
  }

  function renderRooflineReference() {
    const selectedPrecision = rooflinePrecision.value;
    const secondaryPrecision = selectedPrecision === "fp32" ? "fp16" : "fp32";
    const specs = meta.roofline_specs.filter((spec) => rooflineDevice.value === "all" || spec.device === rooflineDevice.value);
    rooflineSpecGridNode.innerHTML = "";
    rooflineReferencePreviewNode.innerHTML = "";
    rooflineReferenceCountNode.textContent = String(specs.length);

    if (!specs.length) {
      rooflineReferenceSummaryNode.textContent = "No roofline reference cards match the current device filter.";
      return;
    }

    const scopeLabel =
      rooflineDevice.value === "all"
        ? "All devices"
        : `${specs[0].label}`;
    rooflineReferenceSummaryNode.textContent =
      `${scopeLabel}. ${selectedPrecision.toUpperCase()} roofs at default clocks. Scroll for the hardware cards.`;

    specs.slice(0, 3).forEach((spec) => {
      const pill = document.createElement("div");
      pill.className = "roofline-preview-pill";
      pill.innerHTML = `
        <span>${spec.device}</span>
        <strong>${formatNumber(spec[`peak_${selectedPrecision}_tflops`], 2)} TFLOP/s</strong>
      `;
      rooflineReferencePreviewNode.appendChild(pill);
    });

    if (specs.length > 3) {
      const extra = document.createElement("div");
      extra.className = "roofline-preview-pill roofline-preview-pill-faint";
      extra.innerHTML = `<span>more</span><strong>+${specs.length - 3} GPUs</strong>`;
      rooflineReferencePreviewNode.appendChild(extra);
    }

    specs.forEach((spec) => {
      const card = document.createElement("article");
      card.className = "roofline-spec-card";
      card.innerHTML = `
        <div class="roofline-spec-head">
          <span class="tag">${spec.device}</span>
          <strong>${spec.label}</strong>
        </div>
        <p>${spec.architecture}, compute capability ${spec.compute_capability}.</p>
        <div class="inline-metrics roofline-inline-metrics"></div>
      `;
      const metrics = card.querySelector(".roofline-inline-metrics");
      metrics.append(
        inlineMetric("bandwidth (GB/s)", spec.memory_bandwidth_gbps, 0),
        inlineMetric(`${selectedPrecision.toUpperCase()} roof`, spec[`peak_${selectedPrecision}_tflops`], 2),
        inlineMetric(`${secondaryPrecision.toUpperCase()} roof`, spec[`peak_${secondaryPrecision}_tflops`], 2)
      );
      rooflineSpecGridNode.appendChild(card);
    });
  }

  function syncPerformancePanelHeights() {
    if (!devicePerfPanelNode || !rooflineReferencePanelNode || !rooflineReferenceBodyNode || !rooflineAtlasPanelNode) {
      return;
    }

    if (window.innerWidth <= 1120) {
      rooflineReferencePanelNode.style.height = "";
      rooflineReferenceBodyNode.style.maxHeight = "";
      return;
    }

    const stackNode = devicePerfPanelNode.parentElement;
    const gapValue = stackNode ? window.getComputedStyle(stackNode).rowGap || window.getComputedStyle(stackNode).gap : "20px";
    const gap = Number.parseFloat(gapValue) || 20;
    const atlasHeight = rooflineAtlasPanelNode.getBoundingClientRect().height;
    const overviewHeight = devicePerfPanelNode.getBoundingClientRect().height;
    const targetHeight = Math.max(280, Math.floor(atlasHeight - overviewHeight - gap));

    rooflineReferencePanelNode.style.height = `${targetHeight}px`;
    rooflineReferenceBodyNode.style.maxHeight = `${Math.max(180, targetHeight - 142)}px`;
  }

  function queuePerformancePanelSync() {
    window.requestAnimationFrame(function () {
      window.requestAnimationFrame(syncPerformancePanelHeights);
    });
  }

  function renderReadingGuide() {
    const uniqueCategories = new Set(meta.category_profiled.map((entry) => entry.category)).size;
    readingGuideMetricsNode.append(
      inlineMetric("GPUs", meta.device_summary.length, 0),
      inlineMetric("profiled binaries", meta.inventory.totals.profiled_sources, 0),
      inlineMetric("kernel rows", kernelRows.length, 0),
      inlineMetric("categories", uniqueCategories, 0)
    );
  }

  function renderCitation() {
    if (!citationMetaNode || !citationBibtexNode || !meta.paper) {
      return;
    }
    citationMetaNode.innerHTML = `
      <div class="citation-actions">
        <div>
          <span class="tag">${meta.paper.venue}</span>
          <h3>${meta.paper.title}</h3>
          <p>Reference link and PDF for the paper that launched this project direction.</p>
        </div>
        <div class="hero-actions" style="margin-top:0;">
          <a class="button secondary" href="${meta.paper.doi_url}" target="_blank" rel="noreferrer">Conference entry</a>
          <a class="button primary" href="${meta.paper.pdf_url}" target="_blank" rel="noreferrer">${meta.paper.pdf_label}</a>
        </div>
      </div>
    `;
    citationBibtexNode.textContent = meta.paper.bibtex;
  }

  function renderTeam(team) {
    if (!teamGridNode || !Array.isArray(team) || !team.length) {
      return;
    }
    team.forEach((member) => {
      const card = document.createElement("article");
      card.className = "team-card";
      card.innerHTML = `
        <img class="team-photo" src="${member.image_path}" alt="${member.name}">
        <div class="team-copy">
          <h3>${member.name}</h3>
          <p>${member.affiliation}</p>
          <a class="button secondary" href="${member.profile_url}" target="_blank" rel="noreferrer">Profile</a>
        </div>
      `;
      teamGridNode.appendChild(card);
    });
  }

  function renderPlot(node, traces, layout, emptyMessage) {
    if (!hasPlotly) {
      emptyState(node, emptyMessage || "Interactive charts require Plotly to load.");
      return;
    }
    window.Plotly.react(node, traces, layout, { responsive: true, displayModeBar: false });
  }

  function renderModelCoverage(modelMatrix) {
    const focusedModels = modelMatrix.filter((entry) => ["cuda", "omp"].includes(entry.model));
    renderPlot(
      modelCoverageNode,
      [
        {
          type: "bar",
          name: "declared sources",
          x: focusedModels.map((entry) => entry.model.toUpperCase()),
          y: focusedModels.map((entry) => entry.available),
          marker: { color: focusedModels.map((entry) => COLORS[entry.model] || "#90b7ff") },
        },
        {
          type: "bar",
          name: "profiled sources",
          x: focusedModels.map((entry) => entry.model.toUpperCase()),
          y: focusedModels.map((entry) => entry.profiled),
          marker: { color: "rgba(255, 156, 91, 0.84)" },
        },
      ],
      basePlotlyLayout({ barmode: "group", yaxis: { title: "source binaries" } })
    );
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

  function renderDevicePerf(devices) {
    renderPlot(
      devicePerfNode,
      [
        {
          type: "bar",
          name: "sources",
          x: devices.map((device) => device.device),
          y: devices.map((device) => device.sources),
          marker: { color: "rgba(144, 183, 255, 0.76)" },
          yaxis: "y",
        },
        {
          type: "scatter",
          mode: "lines+markers",
          name: "median performance",
          x: devices.map((device) => device.device),
          y: devices.map((device) => device.median_performance_tflops),
          marker: { color: "#ff9c5b", size: 10 },
          line: { color: "#ff9c5b", width: 3 },
          yaxis: "y2",
        },
      ],
      basePlotlyLayout({
        yaxis: { title: "source count" },
        yaxis2: {
          title: "median performance (TFLOP/s)",
          overlaying: "y",
          side: "right",
          gridcolor: "rgba(0,0,0,0)",
          color: "#ff9c5b",
        },
      })
    );
    queuePerformancePanelSync();
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
    renderRooflineDetails(rows);
    renderRooflineReference();

    if (!subset.length) {
      emptyState(rooflineNode, "No floating-point kernel rows match the current filters.");
      rooflineSummaryNode.textContent = "";
      queuePerformancePanelSync();
      return;
    }

    const selectedPrecision = rooflinePrecision.value;
    const range = buildRooflineRange(subset);
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
          "AI=%{x:.4f}<br>" +
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
        xaxis: { title: "arithmetic intensity (FLOPs / byte)", type: "log" },
        yaxis: { title: "performance (TFLOP/s)", type: "log" },
        margin: { l: 64, r: 28, t: 30, b: 58 },
      })
    );

    const perfValues = subset.map((row) => Number(row.performance_tflops)).sort((left, right) => left - right);
    const aiValues = subset.map((row) => Number(row.arithmetic_intensity)).sort((left, right) => left - right);
    rooflineSummaryNode.innerHTML = `
      <strong>${subset.length}</strong> floating-point kernel rows in view.
      Median AI <strong>${formatNumber(aiValues[Math.floor(aiValues.length / 2)], 4)}</strong>,
      median performance <strong>${formatNumber(perfValues[Math.floor(perfValues.length / 2)], 4)} TFLOP/s</strong>.
      Dashed lines show the ${selectedPrecision.toUpperCase()} theoretical roofline at default device clocks.
    `;
    queuePerformancePanelSync();
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

  async function loadExplorerDataset() {
    if (!explorerDatasetPromise) {
      explorerDatasetPromise = fetch("./downloads/gpuFLOPBench.json.gz")
        .then((response) => {
          if (!response.ok) {
            throw new Error(`Failed to fetch gpuFLOPBench.json.gz (${response.status}).`);
          }
          if (!response.body || typeof window.DecompressionStream !== "function") {
            throw new Error("This browser cannot decompress the bundled gpuFLOPBench.json.gz dataset.");
          }
          const decompressed = response.body.pipeThrough(new window.DecompressionStream("gzip"));
          return new Response(decompressed).text();
        })
        .then((text) => JSON.parse(text));
    }
    return explorerDatasetPromise;
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
    explorerCodeSummaryNode.textContent = "Loading SASS and IMIX from the structured gpuFLOPBench.json.gz dataset.";
    emptyState(explorerSassCodeNode, "Loading kernel SASS...");
    renderExplorerImix(null, "Loading kernel IMIX...");

    try {
      const dataset = await loadExplorerDataset();
      if (renderToken !== explorerRenderToken) {
        return;
      }

      const rawKernel = dataset?.[selectedPair.program]?.kernels?.[selectedPair.kernel_symbol];
      if (!rawKernel) {
        explorerCodeSummaryNode.textContent = "The selected kernel was not found in gpuFLOPBench.json.gz.";
        emptyState(explorerSassCodeNode, "The selected kernel was not found in the bundled dataset.");
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
      explorerCodeSummaryNode.textContent = "The structured gpuFLOPBench.json.gz dataset could not be loaded in this browser.";
      emptyState(explorerSassCodeNode, "Could not load kernel SASS from gpuFLOPBench.json.gz.");
      renderExplorerImix(null, "Could not load kernel IMIX from gpuFLOPBench.json.gz.");
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

  function init() {
    renderHeroMetrics(meta.hero.headline_metrics);
    renderBenchmarkSurfaces();
    renderDeviceCards(meta.device_summary);
    renderDownloads(meta.downloads);
    renderTopList(peakPerfListNode, "Performance leaders", meta.top_lists.performance_sources);
    renderTopList(aiDenseListNode, "AI-dense leaders", meta.top_lists.ai_dense_sources);
    renderReadingGuide();
    renderCitation();
    renderTeam(meta.team);
    renderModelCoverage(meta.model_matrix);
    renderCategoryCoverage(meta.category_profiled);
    renderDevicePerf(meta.device_summary);

    refillSelect(rooflineDevice, uniqueSorted(kernelRows.map((row) => row.device)), "all devices");
    refillSelect(rooflineModel, uniqueSorted(kernelRows.map((row) => row.model_type)), "all models");
    refillSelect(rooflineCategory, uniqueSorted(kernelRows.map((row) => row.category)), "all categories");
    syncRooflineFilters();

    refillSelect(explorerDevice, uniqueSorted(kernelRows.map((row) => row.device)), "all devices");
    refillSelect(explorerModel, uniqueSorted(kernelRows.map((row) => row.model_type)), "all models");
    refillSelect(explorerCategory, uniqueSorted(kernelRows.map((row) => row.category)), "all categories");

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

    renderRoofline(kernelRows);
    renderExplorer(kernelRows);
    lastUpdatedNode.textContent = new Date(meta.audit.generated_at).toLocaleString();
    window.addEventListener("resize", queuePerformancePanelSync);
  }

  init();
})();
