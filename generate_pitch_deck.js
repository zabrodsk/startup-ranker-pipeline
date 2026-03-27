const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "Rockaway Deal Intelligence";
pres.author = "Rockaway Capital";

// Design tokens
const BG = "0D1117";
const ACCENT = "00D26A";
const TEXT_PRIMARY = "FFFFFF";
const TEXT_SECONDARY = "94A3B8";
const CARD_BG = "161B22";
const BORDER = "21262D";
const FONT = "Calibri";

// ─── SLIDE 1 — Title ───────────────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: BG };

  // Top green bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.08,
    fill: { color: ACCENT }, line: { color: ACCENT }
  });
  // Bottom green bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.545, w: 10, h: 0.08,
    fill: { color: ACCENT }, line: { color: ACCENT }
  });

  slide.addText("Rockaway Deal Intelligence", {
    x: 0, y: 1.8, w: 10, h: 0.8,
    fontSize: 48, bold: true, color: TEXT_PRIMARY,
    fontFace: FONT, align: "center", margin: 0
  });

  slide.addText("AI-Powered Investment Analysis Platform", {
    x: 0, y: 2.8, w: 10, h: 0.5,
    fontSize: 22, color: TEXT_SECONDARY,
    fontFace: FONT, align: "center", margin: 0
  });

  slide.addText("From pitch deck to investment decision in minutes, not hours", {
    x: 0, y: 3.5, w: 10, h: 0.45,
    fontSize: 16, italic: true, color: ACCENT,
    fontFace: FONT, align: "center", margin: 0
  });
}

// ─── SLIDE 2 — The Problem ─────────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: BG };

  slide.addText("The Deal Flow Problem", {
    x: 0.5, y: 0.3, w: 9, h: 0.5,
    fontSize: 36, bold: true, color: TEXT_PRIMARY,
    fontFace: FONT, align: "left", margin: 0
  });

  // Green underline accent
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.85, w: 1.5, h: 0.05,
    fill: { color: ACCENT }, line: { color: ACCENT }
  });

  const bullets = [
    "Analysts spend 4–8 hours reviewing each pitch deck manually",
    "VC firms see 1,000+ deals per year — only 1–2% get funded",
    "Inconsistent evaluation criteria across analysts",
    "Critical signals buried in documents, spreadsheets, and founder profiles",
    "No structured evidence trail for investment decisions"
  ];

  const startY = 1.1;
  const spacing = 0.72;
  const circleD = 0.22;
  const circleR = circleD / 2;

  bullets.forEach((text, i) => {
    const cy = startY + i * spacing;
    // Green circle bullet
    slide.addShape(pres.shapes.OVAL, {
      x: 0.5, y: cy, w: circleD, h: circleD,
      fill: { color: ACCENT }, line: { color: ACCENT }
    });
    // Text
    slide.addText(text, {
      x: 0.9, y: cy - 0.03, w: 8.6, h: 0.3,
      fontSize: 14, color: TEXT_PRIMARY,
      fontFace: FONT, align: "left", margin: 0, valign: "middle"
    });
  });
}

// ─── SLIDE 3 — Solution ────────────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: BG };

  slide.addText("AI That Thinks Like Your Best Analyst", {
    x: 0.5, y: 0.3, w: 9, h: 0.5,
    fontSize: 36, bold: true, color: TEXT_PRIMARY,
    fontFace: FONT, align: "left", margin: 0
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.85, w: 1.5, h: 0.05,
    fill: { color: ACCENT }, line: { color: ACCENT }
  });

  const bullets = [
    "Automated first-pass screening of pitch decks, Specter CSVs, and documents",
    "Evidence-grounded analysis — every conclusion backed by source citations",
    "Pro/contra arguments with devil's advocate critique",
    "Composite scoring: Strategy Fit + Team + Upside → invest / not invest",
    "Company Chat: ask follow-up questions grounded in saved evidence"
  ];

  const startY = 1.1;
  const spacing = 0.72;
  const circleD = 0.22;

  bullets.forEach((text, i) => {
    const cy = startY + i * spacing;
    slide.addShape(pres.shapes.OVAL, {
      x: 0.5, y: cy, w: circleD, h: circleD,
      fill: { color: ACCENT }, line: { color: ACCENT }
    });
    slide.addText(text, {
      x: 0.9, y: cy - 0.03, w: 8.6, h: 0.3,
      fontSize: 14, color: TEXT_PRIMARY,
      fontFace: FONT, align: "left", margin: 0, valign: "middle"
    });
  });
}

// ─── SLIDE 4 — Pipeline ────────────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: BG };

  slide.addText("7-Stage AI Pipeline", {
    x: 0.5, y: 0.25, w: 9, h: 0.45,
    fontSize: 36, bold: true, color: TEXT_PRIMARY,
    fontFace: FONT, align: "left", margin: 0
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.75, w: 1.2, h: 0.05,
    fill: { color: ACCENT }, line: { color: ACCENT }
  });

  const steps = [
    { num: "1", name: "Ingest", desc: "PDF, PPTX, Excel, Specter CSV → chunks" },
    { num: "2", name: "Decompose", desc: "Question trees: Company, Market, Product, Team" },
    { num: "3", name: "Answer", desc: "TF-IDF retrieval + LLM Q&A + web search" },
    { num: "4", name: "Generate", desc: "Pro & Contra arguments from evidence" },
    { num: "5", name: "Critique", desc: "Devil's advocate review" },
    { num: "6", name: "Evaluate", desc: "14 criteria scoring + refinement" },
    { num: "7", name: "Rank", desc: "Composite score → invest decision" }
  ];

  const cardW = 1.2;
  const cardH = 2.8;
  const startX = 0.3;
  const cardY = 1.0;
  const gap = 0.1;

  steps.forEach((step, i) => {
    const cx = startX + i * (cardW + gap);

    // Card background
    slide.addShape(pres.shapes.RECTANGLE, {
      x: cx, y: cardY, w: cardW, h: cardH,
      fill: { color: CARD_BG }, line: { color: BORDER, width: 1 }
    });

    // Step number
    slide.addText(step.num, {
      x: cx, y: cardY + 0.15, w: cardW, h: 0.35,
      fontSize: 18, bold: true, color: ACCENT,
      fontFace: FONT, align: "center", margin: 0
    });

    // Step name
    slide.addText(step.name, {
      x: cx, y: cardY + 0.55, w: cardW, h: 0.3,
      fontSize: 11, bold: true, color: TEXT_PRIMARY,
      fontFace: FONT, align: "center", margin: 0
    });

    // Description
    slide.addText(step.desc, {
      x: cx + 0.05, y: cardY + 0.9, w: cardW - 0.1, h: 1.8,
      fontSize: 9, color: TEXT_SECONDARY,
      fontFace: FONT, align: "center", margin: 0, wrap: true, valign: "top"
    });
  });
}

// ─── SLIDE 5 — LLM Routing ─────────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: BG };

  slide.addText("Smart LLM Routing", {
    x: 0.5, y: 0.25, w: 9, h: 0.45,
    fontSize: 36, bold: true, color: TEXT_PRIMARY,
    fontFace: FONT, align: "left", margin: 0
  });

  slide.addText("Best model for each pipeline phase", {
    x: 0.5, y: 0.8, w: 9, h: 0.3,
    fontSize: 16, color: TEXT_SECONDARY,
    fontFace: FONT, align: "left", margin: 0
  });

  // Build table data
  const headerRow = [
    { text: "Phase", options: { bold: true, fontSize: 13, color: "0D1117", fill: { color: ACCENT }, fontFace: FONT, align: "center" } },
    { text: "Model", options: { bold: true, fontSize: 13, color: "0D1117", fill: { color: ACCENT }, fontFace: FONT, align: "center" } },
    { text: "Provider", options: { bold: true, fontSize: 13, color: "0D1117", fill: { color: ACCENT }, fontFace: FONT, align: "center" } },
    { text: "Purpose", options: { bold: true, fontSize: 13, color: "0D1117", fill: { color: ACCENT }, fontFace: FONT, align: "center" } }
  ];

  const dataRows = [
    ["Decomposition", "Gemini 2.5 Pro", "Google", "Structured question breakdown"],
    ["Answering", "Gemini 2.5 Flash", "Google", "Fast, cost-efficient Q&A"],
    ["Generation", "GPT-4o", "OpenAI", "Creative argument writing"],
    ["Evaluation", "o4-mini", "OpenAI", "Rigorous scoring & reasoning"],
    ["Ranking", "GPT-4o", "OpenAI", "Strategic investment judgment"]
  ];

  const tableData = [headerRow];
  dataRows.forEach((row, i) => {
    const rowBg = i % 2 === 0 ? CARD_BG : BG;
    tableData.push(row.map(cell => ({
      text: cell,
      options: { fontSize: 12, color: TEXT_PRIMARY, fill: { color: rowBg }, fontFace: FONT, align: "left" }
    })));
  });

  slide.addTable(tableData, {
    x: 0.5, y: 1.2, w: 9, colW: [2, 2, 1.5, 3.5],
    rowH: 0.55,
    border: { pt: 1, color: BORDER }
  });

  slide.addText("Budget / Balanced / Premium tiers — configurable per phase", {
    x: 0.5, y: 4.9, w: 9, h: 0.3,
    fontSize: 12, italic: true, color: ACCENT,
    fontFace: FONT, align: "left", margin: 0
  });
}

// ─── SLIDE 6 — Integrations ────────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: BG };

  slide.addText("Full Integration Ecosystem", {
    x: 0.5, y: 0.25, w: 9, h: 0.45,
    fontSize: 36, bold: true, color: TEXT_PRIMARY,
    fontFace: FONT, align: "left", margin: 0
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.75, w: 1.5, h: 0.05,
    fill: { color: ACCENT }, line: { color: ACCENT }
  });

  const columns = [
    {
      label: "AI & LLM",
      x: 0.4,
      items: ["OpenAI GPT-4o, o4-mini", "Google Gemini 2.5 Pro", "Google Gemini 2.5 Flash", "Anthropic Claude Haiku", "OpenRouter"]
    },
    {
      label: "Data & Search",
      x: 3.4,
      items: ["Specter (Founder Intelligence)", "Perplexity Sonar (Web Search)", "Brave Search (Fallback)", "PDF, PPTX, Excel, CSV", "Supabase Storage"]
    },
    {
      label: "Infrastructure",
      x: 6.4,
      items: ["Supabase (PostgreSQL)", "LangGraph (Orchestration)", "LangSmith (Tracing)", "Railway (Hosting)", "Docker (Containers)"]
    }
  ];

  const cardW = 2.8;
  const cardH = 3.5;
  const cardY = 1.0;

  columns.forEach(col => {
    // Card background
    slide.addShape(pres.shapes.RECTANGLE, {
      x: col.x, y: cardY, w: cardW, h: cardH,
      fill: { color: CARD_BG }, line: { color: BORDER, width: 1 }
    });

    // Header label
    slide.addText(col.label, {
      x: col.x + 0.15, y: cardY + 0.15, w: cardW - 0.3, h: 0.35,
      fontSize: 14, bold: true, color: ACCENT,
      fontFace: FONT, align: "left", margin: 0
    });

    // Items
    col.items.forEach((item, i) => {
      slide.addText(item, {
        x: col.x + 0.15, y: cardY + 0.6 + i * 0.52, w: cardW - 0.3, h: 0.45,
        fontSize: 13, color: TEXT_PRIMARY,
        fontFace: FONT, align: "left", margin: 0, valign: "middle"
      });
    });
  });
}

// ─── SLIDE 7 — Input Modes ─────────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: BG };

  slide.addText("Three Ways to Analyze Deals", {
    x: 0.5, y: 0.25, w: 9, h: 0.45,
    fontSize: 36, bold: true, color: TEXT_PRIMARY,
    fontFace: FONT, align: "left", margin: 0
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.75, w: 1.5, h: 0.05,
    fill: { color: ACCENT }, line: { color: ACCENT }
  });

  const cards = [
    { num: "1", name: "Pitch Deck Mode", desc: "Upload PDF or PPTX → instant company analysis from documents", x: 0.35 },
    { num: "2", name: "Specter Mode", desc: "Import company + people CSVs → founder signals combined with company data", x: 3.4 },
    { num: "3", name: "Multi-File Mode", desc: "Upload all available documents per company for the most complete analysis", x: 6.45 }
  ];

  const cardW = 2.9;
  const cardH = 3.5;
  const cardY = 1.0;

  cards.forEach(card => {
    slide.addShape(pres.shapes.RECTANGLE, {
      x: card.x, y: cardY, w: cardW, h: cardH,
      fill: { color: CARD_BG }, line: { color: BORDER, width: 1 }
    });

    // Mode number
    slide.addText(card.num, {
      x: card.x, y: cardY + 0.2, w: cardW, h: 0.55,
      fontSize: 36, bold: true, color: ACCENT,
      fontFace: FONT, align: "center", margin: 0
    });

    // Mode name
    slide.addText(card.name, {
      x: card.x + 0.1, y: cardY + 0.9, w: cardW - 0.2, h: 0.45,
      fontSize: 18, bold: true, color: TEXT_PRIMARY,
      fontFace: FONT, align: "center", margin: 0, wrap: true
    });

    // Description
    slide.addText(card.desc, {
      x: card.x + 0.15, y: cardY + 1.5, w: cardW - 0.3, h: 1.8,
      fontSize: 13, color: TEXT_SECONDARY,
      fontFace: FONT, align: "center", margin: 0, wrap: true, valign: "top"
    });
  });
}

// ─── SLIDE 8 — Outputs ─────────────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: BG };

  slide.addText("Structured Investment Intelligence", {
    x: 0.5, y: 0.25, w: 9, h: 0.45,
    fontSize: 36, bold: true, color: TEXT_PRIMARY,
    fontFace: FONT, align: "left", margin: 0
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.75, w: 1.8, h: 0.05,
    fill: { color: ACCENT }, line: { color: ACCENT }
  });

  const leftItems = [
    "Composite score: Strategy Fit + Team + Upside",
    "Triage bucket: Priority Review / Watchlist / Low Priority",
    "Ranked pro/contra arguments with source citations",
    "Excel export: Summary + Arguments + Evidence sheets"
  ];

  const rightItems = [
    "Executive summary + key points + red flags",
    "Company Chat for persistent team Q&A",
    "Full token-level cost tracking per analysis"
  ];

  const circleD = 0.18;
  const startY = 1.1;
  const rowSpacing = 0.68;

  // Left column
  leftItems.forEach((text, i) => {
    const cy = startY + i * rowSpacing;
    slide.addShape(pres.shapes.OVAL, {
      x: 0.5, y: cy, w: circleD, h: circleD,
      fill: { color: ACCENT }, line: { color: ACCENT }
    });
    slide.addText(text, {
      x: 0.85, y: cy - 0.03, w: 4.1, h: 0.3,
      fontSize: 14, color: TEXT_PRIMARY,
      fontFace: FONT, align: "left", margin: 0, wrap: true, valign: "middle"
    });
  });

  // Right column
  rightItems.forEach((text, i) => {
    const cy = startY + i * rowSpacing;
    slide.addShape(pres.shapes.OVAL, {
      x: 5.2, y: cy, w: circleD, h: circleD,
      fill: { color: ACCENT }, line: { color: ACCENT }
    });
    slide.addText(text, {
      x: 5.55, y: cy - 0.03, w: 4.1, h: 0.3,
      fontSize: 14, color: TEXT_PRIMARY,
      fontFace: FONT, align: "left", margin: 0, wrap: true, valign: "middle"
    });
  });
}

// ─── SLIDE 9 — Tech Stack ──────────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: BG };

  slide.addText("Production-Ready Architecture", {
    x: 0.5, y: 0.25, w: 9, h: 0.45,
    fontSize: 36, bold: true, color: TEXT_PRIMARY,
    fontFace: FONT, align: "left", margin: 0
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.75, w: 1.5, h: 0.05,
    fill: { color: ACCENT }, line: { color: ACCENT }
  });

  const cards = [
    {
      x: 0.3,
      label: "Backend & Infrastructure",
      items: ["Python 3.10+ / FastAPI", "LangChain + LangGraph", "Supabase (PostgreSQL + Storage)", "Uvicorn / Railway hosting", "Docker containerization"]
    },
    {
      x: 5.0,
      label: "AI & Processing",
      items: ["Multi-provider LLM routing", "TF-IDF chunk retrieval", "Parallel async pipeline", "Rate limiting + retry logic", "Token cost tracking"]
    }
  ];

  const cardW = 4.4;
  const cardH = 3.5;
  const cardY = 1.0;

  cards.forEach(card => {
    slide.addShape(pres.shapes.RECTANGLE, {
      x: card.x, y: cardY, w: cardW, h: cardH,
      fill: { color: CARD_BG }, line: { color: BORDER, width: 1 }
    });

    slide.addText(card.label, {
      x: card.x + 0.2, y: cardY + 0.15, w: cardW - 0.4, h: 0.35,
      fontSize: 16, bold: true, color: ACCENT,
      fontFace: FONT, align: "left", margin: 0
    });

    card.items.forEach((item, i) => {
      slide.addText(item, {
        x: card.x + 0.2, y: cardY + 0.65 + i * 0.52, w: cardW - 0.4, h: 0.45,
        fontSize: 13, color: TEXT_PRIMARY,
        fontFace: FONT, align: "left", margin: 0, valign: "middle"
      });
    });
  });
}

// ─── SLIDE 10 — Deployment ─────────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: BG };

  slide.addText("Flexible Deployment Options", {
    x: 0.5, y: 0.25, w: 9, h: 0.45,
    fontSize: 36, bold: true, color: TEXT_PRIMARY,
    fontFace: FONT, align: "left", margin: 0
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.75, w: 1.5, h: 0.05,
    fill: { color: ACCENT }, line: { color: ACCENT }
  });

  const deployCards = [
    { name: "Railway Cloud", desc: "Production deployment: web service + dedicated Specter worker. Auto-scaling, managed.", x: 0.35, y: 1.05 },
    { name: "Local Development", desc: "Uvicorn + FastAPI with optional Supabase. Full feature parity.", x: 4.85, y: 1.05 },
    { name: "Cloudflare Tunnel", desc: "Lightweight team sharing via slim share. No infrastructure needed.", x: 0.35, y: 3.05 },
    { name: "Docker", desc: "Containerized anywhere. Consistent environment across all deployments.", x: 4.85, y: 3.05 }
  ];

  const cardW = 4.3;
  const cardH = 1.8;

  deployCards.forEach(card => {
    slide.addShape(pres.shapes.RECTANGLE, {
      x: card.x, y: card.y, w: cardW, h: cardH,
      fill: { color: CARD_BG }, line: { color: BORDER, width: 1 }
    });

    slide.addText(card.name, {
      x: card.x + 0.2, y: card.y + 0.15, w: cardW - 0.4, h: 0.35,
      fontSize: 15, bold: true, color: ACCENT,
      fontFace: FONT, align: "left", margin: 0
    });

    slide.addText(card.desc, {
      x: card.x + 0.2, y: card.y + 0.6, w: cardW - 0.4, h: 1.0,
      fontSize: 13, color: TEXT_SECONDARY,
      fontFace: FONT, align: "left", margin: 0, wrap: true, valign: "top"
    });
  });
}

// ─── SLIDE 11 — Company Chat ───────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: BG };

  slide.addText("Company Chat", {
    x: 0.5, y: 0.25, w: 9, h: 0.45,
    fontSize: 36, bold: true, color: TEXT_PRIMARY,
    fontFace: FONT, align: "left", margin: 0
  });

  slide.addText("Your AI Research Assistant — Always On", {
    x: 0.5, y: 0.8, w: 9, h: 0.35,
    fontSize: 20, color: ACCENT,
    fontFace: FONT, align: "left", margin: 0
  });

  const features = [
    { label: "Evidence-Grounded", desc: "Answers anchored in saved analysis evidence" },
    { label: "Web Search Fallback", desc: "Broader context via Perplexity when needed" },
    { label: "Team-Shared", desc: "Persistent history across all users via Supabase" },
    { label: "Model Selection", desc: "Choose LLM model per session" },
    { label: "Cost Visibility", desc: "Per-answer token cost shown inline" },
    { label: "No Re-Analysis", desc: "Ask questions without rerunning the pipeline" }
  ];

  // 2 columns, 3 rows
  const col1X = 0.35;
  const col2X = 5.0;
  const rowYs = [1.4, 2.5, 3.6];
  const itemW = 4.2;
  const itemH = 0.9;
  const circleD = 0.16;

  features.forEach((feat, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = col === 0 ? col1X : col2X;
    const y = rowYs[row];

    // Small green circle
    slide.addShape(pres.shapes.OVAL, {
      x: x, y: y + 0.05, w: circleD, h: circleD,
      fill: { color: ACCENT }, line: { color: ACCENT }
    });

    // Bold label
    slide.addText(feat.label, {
      x: x + 0.25, y: y, w: itemW - 0.25, h: 0.3,
      fontSize: 14, bold: true, color: TEXT_PRIMARY,
      fontFace: FONT, align: "left", margin: 0
    });

    // Description
    slide.addText(feat.desc, {
      x: x + 0.25, y: y + 0.32, w: itemW - 0.25, h: 0.5,
      fontSize: 13, color: TEXT_SECONDARY,
      fontFace: FONT, align: "left", margin: 0, wrap: true
    });
  });
}

// ─── SLIDE 12 — Closing / Vision ───────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: BG };

  // Top green bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.08,
    fill: { color: ACCENT }, line: { color: ACCENT }
  });

  // Bottom green bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.545, w: 10, h: 0.08,
    fill: { color: ACCENT }, line: { color: ACCENT }
  });

  slide.addText("The Future of Deal Intelligence", {
    x: 0, y: 1.2, w: 10, h: 0.7,
    fontSize: 40, bold: true, color: TEXT_PRIMARY,
    fontFace: FONT, align: "center", margin: 0
  });

  slide.addText("Every VC firm deserves an AI analyst that never sleeps", {
    x: 0, y: 2.1, w: 10, h: 0.45,
    fontSize: 20, italic: true, color: ACCENT,
    fontFace: FONT, align: "center", margin: 0
  });

  const bullets = [
    "Scale from 10 to 1,000+ deals per year without growing the team",
    "Consistent, evidence-backed evaluation across all analysts",
    "Full audit trail of every investment decision",
    "Built by Rockaway Capital — for the way VCs actually work"
  ];

  const startY = 2.9;
  const spacing = 0.55;
  const circleD = 0.18;
  const groupX = 1.0;

  bullets.forEach((text, i) => {
    const cy = startY + i * spacing;
    slide.addShape(pres.shapes.OVAL, {
      x: groupX, y: cy, w: circleD, h: circleD,
      fill: { color: ACCENT }, line: { color: ACCENT }
    });
    slide.addText(text, {
      x: groupX + 0.35, y: cy - 0.02, w: 8.5, h: 0.3,
      fontSize: 15, color: TEXT_PRIMARY,
      fontFace: FONT, align: "left", margin: 0, valign: "middle"
    });
  });
}

// ─── Write file ────────────────────────────────────────────────────────────
const outputPath = "/Users/dusanzabrodsky/Library/Mobile Documents/com~apple~CloudDocs/coding/Rockaway Deal Intelligence/Rockaway_Deal_Intelligence_Pitch_Deck.pptx";

pres.writeFile({ fileName: outputPath })
  .then(() => console.log("SUCCESS: Pitch deck written to", outputPath))
  .catch(err => { console.error("ERROR:", err); process.exit(1); });
