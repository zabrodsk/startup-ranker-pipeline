const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "Rockaway Deal Intelligence Pitch Deck";
pres.author = "Rockaway Capital";

// Brand Colors (no # prefix)
const C = {
  dark:      "2C3337",
  light:     "E2E2E2",
  green:     "4CAF82",
  white:     "FFFFFF",
  grayText:  "A8B5C0",
  cardBg:    "3A4248",
};

// ─────────────────────────────────────────────
// SLIDE 1 — Titulní slide
// ─────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: C.dark };

  // Top accent bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.07,
    fill: { color: C.green }, line: { color: C.green, width: 0 }
  });
  // Bottom accent bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.555, w: 10, h: 0.07,
    fill: { color: C.green }, line: { color: C.green, width: 0 }
  });

  // ROCKAWAY
  slide.addText("ROCKAWAY", {
    x: 0, y: 0.8, w: 10, h: 0.4,
    align: "center", fontSize: 14, bold: true, color: C.green, margin: 0
  });

  // DEAL INTELLIGENCE
  slide.addText("DEAL INTELLIGENCE", {
    x: 0, y: 1.3, w: 10, h: 0.8,
    align: "center", fontSize: 44, bold: true, color: C.white, margin: 0
  });

  // Divider line
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.5, y: 2.5, w: 3, h: 0.03,
    fill: { color: C.green }, line: { color: C.green, width: 0 }
  });

  // Subtitle
  slide.addText("AI platforma pro investiční analýzu", {
    x: 0, y: 2.8, w: 10, h: 0.45,
    align: "center", fontSize: 20, color: C.grayText, margin: 0
  });

  // Tagline
  slide.addText("Od pitch decku k investičnímu rozhodnutí za minuty, ne hodiny", {
    x: 0, y: 3.5, w: 10, h: 0.4,
    align: "center", fontSize: 14, italic: true, color: C.green, margin: 0
  });
}

// ─────────────────────────────────────────────
// SLIDE 2 — Problém
// ─────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: C.dark };

  slide.addText("Problém: Přetížení deal flow", {
    x: 0.5, y: 0.3, w: 9, h: 0.5,
    fontSize: 34, bold: true, color: C.white, margin: 0
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.82, w: 1.8, h: 0.05,
    fill: { color: C.green }, line: { color: C.green, width: 0 }
  });

  const bullets = [
    "Analytici tráví 4–8 hodin manuálním hodnocením každého pitch decku",
    "VC fondy obdrží 1 000+ dealů ročně — financují jen 1–2 %",
    "Nekonzistentní hodnotící kritéria napříč analytiky",
    "Klíčové signály ukryté v dokumentech, tabulkách a profilech zakladatelů",
    "Žádný strukturovaný záznam důkazů pro investiční rozhodnutí",
  ];

  const startY = 1.1;
  const spacing = 0.72;
  const circleSize = 0.2;

  bullets.forEach((text, i) => {
    const y = startY + i * spacing;
    // Circle bullet
    slide.addShape(pres.shapes.OVAL, {
      x: 0.5, y: y + 0.02, w: circleSize, h: circleSize,
      fill: { color: C.green }, line: { color: C.green, width: 0 }
    });
    // Text
    slide.addText(text, {
      x: 0.85, y: y, w: 8.8, h: 0.5,
      fontSize: 14, color: C.light, margin: 0, valign: "middle"
    });
  });
}

// ─────────────────────────────────────────────
// SLIDE 3 — Řešení
// ─────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: C.dark };

  slide.addText("Řešení: AI, které myslí jako váš nejlepší analytik", {
    x: 0.5, y: 0.3, w: 9, h: 0.5,
    fontSize: 30, bold: true, color: C.white, margin: 0
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.82, w: 2.5, h: 0.05,
    fill: { color: C.green }, line: { color: C.green, width: 0 }
  });

  const bullets = [
    "Automatizované první hodnocení pitch decků, Specter CSV a dokumentů",
    "Analýza podložená důkazy — každý závěr cituje zdrojový dokument",
    "Pro/contra argumenty s devil's advocate kritikou",
    "Kompozitní skóre: Strategický fit + Tým + Potenciál → investovat / neinvestovat",
    "Firemní chat: doptejte se na detaily podložené uloženými důkazy",
  ];

  const startY = 1.1;
  const spacing = 0.72;
  const circleSize = 0.2;

  bullets.forEach((text, i) => {
    const y = startY + i * spacing;
    slide.addShape(pres.shapes.OVAL, {
      x: 0.5, y: y + 0.02, w: circleSize, h: circleSize,
      fill: { color: C.green }, line: { color: C.green, width: 0 }
    });
    slide.addText(text, {
      x: 0.85, y: y, w: 8.8, h: 0.5,
      fontSize: 14, color: C.light, margin: 0, valign: "middle"
    });
  });
}

// ─────────────────────────────────────────────
// SLIDE 4 — Pipeline (7 kroků)
// ─────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: C.dark };

  slide.addText("7fázový AI pipeline", {
    x: 0.5, y: 0.2, w: 9, h: 0.5,
    fontSize: 34, bold: true, color: C.white, margin: 0
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.72, w: 1.2, h: 0.05,
    fill: { color: C.green }, line: { color: C.green, width: 0 }
  });

  const steps = [
    { num: "1", name: "Ingest", desc: "PDF, PPTX, Excel, Specter CSV → bloky textu" },
    { num: "2", name: "Dekompozice", desc: "Strom otázek: Firma, Trh, Produkt, Tým" },
    { num: "3", name: "Odpovědi", desc: "TF-IDF retrieval + LLM + webové vyhledávání" },
    { num: "4", name: "Generování", desc: "Pro & Contra argumenty ze všech důkazů" },
    { num: "5", name: "Kritika", desc: "Devil's advocate recenze každého argumentu" },
    { num: "6", name: "Hodnocení", desc: "14 kritérií + iterativní zdokonalování" },
    { num: "7", name: "Ranking", desc: "Kompozitní skóre → investiční rozhodnutí" },
  ];

  const cardW = 1.2;
  const cardH = 3.4;
  const cardTopY = 0.95;
  const xPositions = [0.3, 1.6, 2.9, 4.2, 5.5, 6.8, 8.1];

  steps.forEach((step, i) => {
    const x = xPositions[i];
    // Card background
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y: cardTopY, w: cardW, h: cardH,
      fill: { color: C.cardBg }, line: { color: C.cardBg, width: 0 }
    });
    // Step number
    slide.addText(step.num, {
      x, y: cardTopY + 0.12, w: cardW, h: 0.4,
      align: "center", fontSize: 18, bold: true, color: C.green, margin: 0
    });
    // Step name
    slide.addText(step.name, {
      x: x + 0.05, y: cardTopY + 0.55, w: cardW - 0.1, h: 0.55,
      align: "center", fontSize: 11, bold: true, color: C.white,
      wrap: true, margin: 0
    });
    // Description
    slide.addText(step.desc, {
      x: x + 0.05, y: cardTopY + 1.1, w: cardW - 0.1, h: 2.1,
      align: "center", fontSize: 8, color: C.grayText,
      wrap: true, margin: 0, valign: "top"
    });
  });
}

// ─────────────────────────────────────────────
// SLIDE 5 — LLM routing
// ─────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: C.dark };

  slide.addText("Chytré přiřazení modelů — nejlepší AI pro každý úkol", {
    x: 0.5, y: 0.25, w: 9, h: 0.55,
    fontSize: 28, bold: true, color: C.white, margin: 0
  });

  slide.addText("Multi-provider architektura: Google · OpenAI · Anthropic · OpenRouter", {
    x: 0.5, y: 0.8, w: 9, h: 0.35,
    fontSize: 14, color: C.green, margin: 0
  });

  // Table
  const colWidths = [2, 2.2, 1.8, 3.2];
  const headerRow = [
    { text: "Fáze",          options: { fill: { color: C.green }, color: C.dark, bold: true, fontSize: 13, align: "center" } },
    { text: "Model",         options: { fill: { color: C.green }, color: C.dark, bold: true, fontSize: 13, align: "center" } },
    { text: "Poskytovatel",  options: { fill: { color: C.green }, color: C.dark, bold: true, fontSize: 13, align: "center" } },
    { text: "Účel",          options: { fill: { color: C.green }, color: C.dark, bold: true, fontSize: 13, align: "center" } },
  ];

  const dataRows = [
    ["Dekompozice",  "Gemini 3.1 Pro",     "Google", "Strukturované rozdělení otázek"],
    ["Odpovídání",   "Gemini 2.5 Flash",   "Google", "Rychlé a cenově efektivní Q&A"],
    ["Generování",   "GPT-5.2",            "OpenAI", "Tvorba přesvědčivých argumentů"],
    ["Hodnocení",    "o4-mini",            "OpenAI", "Přísné skórování a logické uvažování"],
    ["Ranking",      "GPT-5.2",            "OpenAI", "Strategický investiční úsudek"],
  ];

  const tableData = [headerRow];
  dataRows.forEach((row, i) => {
    const bg = i % 2 === 0 ? C.cardBg : C.dark;
    tableData.push(
      row.map(cell => ({
        text: cell,
        options: { fill: { color: bg }, color: C.light, fontSize: 13, align: "center", valign: "middle" }
      }))
    );
  });

  slide.addTable(tableData, {
    x: 0.4, y: 1.15, w: 9.2,
    rowH: 0.53,
    colW: colWidths,
    border: { pt: 0 },
  });

  slide.addText("Rozpočtová / Vyvážená / Prémiová úroveň — konfigurovatelná pro každou fázi", {
    x: 0.5, y: 4.85, w: 9, h: 0.35,
    fontSize: 11, italic: true, color: C.green, margin: 0
  });
}

// ─────────────────────────────────────────────
// SLIDE 6 — Integrace
// ─────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: C.dark };

  slide.addText("Kompletní integrační ekosystém", {
    x: 0.5, y: 0.25, w: 9, h: 0.5,
    fontSize: 34, bold: true, color: C.white, margin: 0
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.78, w: 2.0, h: 0.05,
    fill: { color: C.green }, line: { color: C.green, width: 0 }
  });

  const cards = [
    {
      title: "AI & Jazykové modely",
      items: ["OpenAI GPT-5, o4-mini", "Google Gemini 3.1 Pro", "Google Gemini 2.5 Flash", "Anthropic Claude Haiku", "OpenRouter"]
    },
    {
      title: "Data & Vyhledávání",
      items: ["Specter (signály zakladatelů)", "Perplexity Sonar (webový search)", "Brave Search (záloha)", "PDF, PPTX, Excel, CSV", "Supabase Storage"]
    },
    {
      title: "Infrastruktura",
      items: ["Supabase (PostgreSQL)", "LangGraph (orchestrace)", "LangSmith (trasování)", "Railway (produkce)", "Docker (kontejnery)"]
    }
  ];

  const cardW = 2.8;
  const cardH = 3.6;
  const cardY = 1.0;
  const xPos = [0.4, 3.4, 6.4];

  cards.forEach((card, i) => {
    const x = xPos[i];
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y: cardY, w: cardW, h: cardH,
      fill: { color: C.cardBg }, line: { color: C.cardBg, width: 0 }
    });
    // Header
    slide.addText(card.title, {
      x: x + 0.2, y: cardY + 0.15, w: cardW - 0.3, h: 0.4,
      fontSize: 15, bold: true, color: C.green, margin: 0
    });
    // Items
    const itemText = card.items.map((item, j) => ({
      text: item,
      options: { breakLine: j < card.items.length - 1, paraSpaceAfter: 6 }
    }));
    slide.addText(itemText, {
      x: x + 0.2, y: cardY + 0.65, w: cardW - 0.3, h: cardH - 0.75,
      fontSize: 12.5, color: C.light, margin: 0, valign: "top"
    });
  });
}

// ─────────────────────────────────────────────
// SLIDE 7 — Vstupní módy
// ─────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: C.dark };

  slide.addText("Tři způsoby analýzy dealů", {
    x: 0.5, y: 0.25, w: 9, h: 0.5,
    fontSize: 34, bold: true, color: C.white, margin: 0
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.78, w: 1.8, h: 0.05,
    fill: { color: C.green }, line: { color: C.green, width: 0 }
  });

  const modes = [
    { num: "1", name: "Pitch Deck mód",   desc: "Nahrajte PDF nebo PPTX → okamžitá analýza firmy z dokumentů" },
    { num: "2", name: "Specter mód",      desc: "Import CSV z firem a lidí ze Specteru → signály zakladatelů + data o firmě" },
    { num: "3", name: "Multi-file mód",   desc: "Nahrajte všechny dostupné dokumenty → nejkomplexnější analýza" },
  ];

  const cardW = 2.9;
  const cardH = 3.6;
  const cardY = 1.0;
  const xPos = [0.35, 3.4, 6.45];

  modes.forEach((mode, i) => {
    const x = xPos[i];
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y: cardY, w: cardW, h: cardH,
      fill: { color: C.cardBg }, line: { color: C.cardBg, width: 0 }
    });
    // Large number
    slide.addText(mode.num, {
      x, y: cardY + 0.15, w: cardW, h: 0.7,
      align: "center", fontSize: 40, bold: true, color: C.green, margin: 0
    });
    // Mode name
    slide.addText(mode.name, {
      x: x + 0.1, y: cardY + 0.8, w: cardW - 0.2, h: 0.5,
      align: "center", fontSize: 18, bold: true, color: C.white, margin: 0
    });
    // Description
    slide.addText(mode.desc, {
      x: x + 0.15, y: cardY + 1.3, w: cardW - 0.3, h: 2.1,
      align: "center", fontSize: 13, color: C.grayText, wrap: true, margin: 0, valign: "top"
    });
  });
}

// ─────────────────────────────────────────────
// SLIDE 8 — Výstupy
// ─────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: C.dark };

  slide.addText("Strukturovaná investiční inteligence", {
    x: 0.5, y: 0.25, w: 9, h: 0.5,
    fontSize: 34, bold: true, color: C.white, margin: 0
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.78, w: 2.0, h: 0.05,
    fill: { color: C.green }, line: { color: C.green, width: 0 }
  });

  const leftItems = [
    "Kompozitní skóre: Strategický fit + Tým + Potenciál",
    "Třídění: Prioritní review / Watchlist / Nízká priorita",
    "Seřazené pro/contra argumenty s citacemi zdrojů",
    "Export do Excelu: Souhrn + Argumenty + Důkazy",
  ];
  const rightItems = [
    "Exekutivní souhrn + klíčové body + red flags",
    "Firemní chat pro trvalé Q&A celého týmu",
    "Sledování nákladů na úrovni tokenů",
  ];

  const startY = 1.1;
  const rowSpacing = 0.7;
  const circleSize = 0.18;

  leftItems.forEach((text, i) => {
    const y = startY + i * rowSpacing;
    slide.addShape(pres.shapes.OVAL, {
      x: 0.4, y: y + 0.05, w: circleSize, h: circleSize,
      fill: { color: C.green }, line: { color: C.green, width: 0 }
    });
    slide.addText(text, {
      x: 0.72, y: y, w: 4.0, h: 0.5,
      fontSize: 14, color: C.light, margin: 0, valign: "middle", wrap: true
    });
  });

  rightItems.forEach((text, i) => {
    const y = startY + i * rowSpacing;
    slide.addShape(pres.shapes.OVAL, {
      x: 5.1, y: y + 0.05, w: circleSize, h: circleSize,
      fill: { color: C.green }, line: { color: C.green, width: 0 }
    });
    slide.addText(text, {
      x: 5.42, y: y, w: 4.2, h: 0.5,
      fontSize: 14, color: C.light, margin: 0, valign: "middle", wrap: true
    });
  });
}

// ─────────────────────────────────────────────
// SLIDE 9 — Technický stack
// ─────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: C.dark };

  slide.addText("Produkčně připravená architektura", {
    x: 0.5, y: 0.25, w: 9, h: 0.5,
    fontSize: 34, bold: true, color: C.white, margin: 0
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.78, w: 1.8, h: 0.05,
    fill: { color: C.green }, line: { color: C.green, width: 0 }
  });

  const cards = [
    {
      x: 0.3, title: "Backend & Infrastruktura",
      items: ["Python 3.10+ / FastAPI", "LangChain + LangGraph", "Supabase (PostgreSQL + Storage)", "Uvicorn / Railway hosting", "Docker kontejnerizace"]
    },
    {
      x: 5.0, title: "AI & Zpracování",
      items: ["Multi-provider LLM routing", "TF-IDF chunk retrieval", "Paralelní async pipeline", "Rate limiting + retry logika", "Sledování nákladů tokenů"]
    }
  ];

  cards.forEach(card => {
    slide.addShape(pres.shapes.RECTANGLE, {
      x: card.x, y: 1.0, w: 4.4, h: 3.6,
      fill: { color: C.cardBg }, line: { color: C.cardBg, width: 0 }
    });
    slide.addText(card.title, {
      x: card.x + 0.2, y: 1.15, w: 4.0, h: 0.4,
      fontSize: 15, bold: true, color: C.green, margin: 0
    });
    const itemText = card.items.map((item, j) => ({
      text: item,
      options: { breakLine: j < card.items.length - 1, paraSpaceAfter: 8 }
    }));
    slide.addText(itemText, {
      x: card.x + 0.2, y: 1.65, w: 4.0, h: 2.8,
      fontSize: 13, color: C.light, margin: 0, valign: "top"
    });
  });
}

// ─────────────────────────────────────────────
// SLIDE 10 — Nasazení
// ─────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: C.dark };

  slide.addText("Flexibilní možnosti nasazení", {
    x: 0.5, y: 0.25, w: 9, h: 0.5,
    fontSize: 34, bold: true, color: C.white, margin: 0
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.78, w: 1.8, h: 0.05,
    fill: { color: C.green }, line: { color: C.green, width: 0 }
  });

  const deployCards = [
    { name: "Railway Cloud",     desc: "Produkce: webová služba + dedicated Specter worker. Automatické škálování.", x: 0.35, y: 1.05 },
    { name: "Lokální vývoj",     desc: "Uvicorn + FastAPI s volitelnou Supabase. Plná parita funkcí.",               x: 4.9,  y: 1.05 },
    { name: "Cloudflare Tunnel", desc: "Lehké sdílení týmu přes slim share. Bez infrastruktury.",                    x: 0.35, y: 3.1  },
    { name: "Docker",            desc: "Kontejnerizováno kdekoliv. Konzistentní prostředí napříč nasazeními.",       x: 4.9,  y: 3.1  },
  ];

  deployCards.forEach(card => {
    slide.addShape(pres.shapes.RECTANGLE, {
      x: card.x, y: card.y, w: 4.3, h: 1.85,
      fill: { color: C.cardBg }, line: { color: C.cardBg, width: 0 }
    });
    slide.addText(card.name, {
      x: card.x + 0.2, y: card.y + 0.15, w: 3.9, h: 0.4,
      fontSize: 15, bold: true, color: C.green, margin: 0
    });
    slide.addText(card.desc, {
      x: card.x + 0.2, y: card.y + 0.6, w: 3.9, h: 1.1,
      fontSize: 12, color: C.grayText, wrap: true, margin: 0, valign: "top"
    });
  });
}

// ─────────────────────────────────────────────
// SLIDE 11 — Firemní chat
// ─────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: C.dark };

  slide.addText("Firemní chat", {
    x: 0.5, y: 0.25, w: 9, h: 0.55,
    fontSize: 36, bold: true, color: C.white, margin: 0
  });

  slide.addText("Váš AI výzkumný asistent — vždy dostupný", {
    x: 0.5, y: 0.8, w: 9, h: 0.4,
    fontSize: 18, color: C.green, margin: 0
  });

  // 2×3 grid of features
  const features = [
    // col1                                         col2
    { label: "Podložené důkazy", desc: "Odpovědi zakotvené v uložených datech analýzy",         col: 0, row: 0 },
    { label: "Webové vyhledávání", desc: "Širší kontext přes Perplexity, když chybí lokální data", col: 1, row: 0 },
    { label: "Sdíleno v týmu", desc: "Trvalá historie pro všechny uživatele přes Supabase",    col: 0, row: 1 },
    { label: "Výběr modelu", desc: "Volba LLM modelu pro každou session",                      col: 1, row: 1 },
    { label: "Viditelné náklady", desc: "Náklady na tokeny zobrazeny u každé odpovědi",          col: 0, row: 2 },
    { label: "Bez re-analýzy", desc: "Otázky bez nutnosti spouštět pipeline znovu",            col: 1, row: 2 },
  ];

  const col1X = 0.35;
  const col2X = 5.1;
  const rowYs = [1.45, 2.55, 3.65];
  const circleSize = 0.18;

  features.forEach(feat => {
    const x = feat.col === 0 ? col1X : col2X;
    const y = rowYs[feat.row];
    // Circle
    slide.addShape(pres.shapes.OVAL, {
      x, y: y + 0.05, w: circleSize, h: circleSize,
      fill: { color: C.green }, line: { color: C.green, width: 0 }
    });
    // Label (bold)
    slide.addText(feat.label, {
      x: x + 0.27, y: y, w: 4.3, h: 0.3,
      fontSize: 14, bold: true, color: C.white, margin: 0
    });
    // Description
    slide.addText(feat.desc, {
      x: x + 0.27, y: y + 0.3, w: 4.3, h: 0.4,
      fontSize: 12, color: C.grayText, margin: 0, wrap: true
    });
  });
}

// ─────────────────────────────────────────────
// SLIDE 12 — Závěr
// ─────────────────────────────────────────────
{
  const slide = pres.addSlide();
  slide.background = { color: C.dark };

  // Top green bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.07,
    fill: { color: C.green }, line: { color: C.green, width: 0 }
  });
  // Bottom green bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.555, w: 10, h: 0.07,
    fill: { color: C.green }, line: { color: C.green, width: 0 }
  });

  // ROCKAWAY
  slide.addText("ROCKAWAY", {
    x: 0, y: 0.9, w: 10, h: 0.4,
    align: "center", fontSize: 14, bold: true, color: C.green, margin: 0
  });

  // Title
  slide.addText("Budoucnost investiční inteligence", {
    x: 0, y: 1.4, w: 10, h: 0.7,
    align: "center", fontSize: 40, bold: true, color: C.white, margin: 0
  });

  // Divider
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.5, y: 2.45, w: 3, h: 0.03,
    fill: { color: C.green }, line: { color: C.green, width: 0 }
  });

  // Tagline
  slide.addText("Každý VC fond si zaslouží AI analytika, který nikdy nespí", {
    x: 0, y: 2.65, w: 10, h: 0.45,
    align: "center", fontSize: 17, italic: true, color: C.green, margin: 0
  });

  // 4 bullets
  const closingBullets = [
    "Škálujte ze 10 na 1 000+ dealů ročně bez rozšiřování týmu",
    "Konzistentní hodnocení podložené důkazy napříč všemi analytiky",
    "Kompletní auditní stopa každého investičního rozhodnutí",
    "Vyvinuto Rockaway Capital — pro způsob, jakým VCs skutečně pracují",
  ];

  const startY = 3.2;
  const spacing = 0.52;
  const circleSize = 0.18;

  closingBullets.forEach((text, i) => {
    const y = startY + i * spacing;
    // For centering group across 1.0"–9.0" (8" wide)
    slide.addShape(pres.shapes.OVAL, {
      x: 1.0, y: y + 0.05, w: circleSize, h: circleSize,
      fill: { color: C.green }, line: { color: C.green, width: 0 }
    });
    slide.addText(text, {
      x: 1.28, y: y, w: 7.7, h: 0.4,
      fontSize: 14, color: C.light, margin: 0, valign: "middle"
    });
  });
}

// ─────────────────────────────────────────────
// Write file
// ─────────────────────────────────────────────
const outputPath = "/Users/dusanzabrodsky/Library/Mobile Documents/com~apple~CloudDocs/coding/Rockaway Deal Intelligence/Rockaway_Deal_Intelligence_Pitch_Deck_CZ.pptx";

pres.writeFile({ fileName: outputPath })
  .then(() => console.log("✅ Pitch deck generated:", outputPath))
  .catch(err => { console.error("❌ Error:", err); process.exit(1); });
