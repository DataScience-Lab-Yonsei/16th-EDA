import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "C:/Users/zzoll/OneDrive/문서/EDA project - 법률";
const TMP = "C:/Users/zzoll/AppData/Local/Temp/codex-presentations/manual-eda-law/judgment-style/tmp";
const OUT = `${ROOT}/outputs/judgment-style-diachronic-analysis.pptx`;
const PREVIEW = `${TMP}/preview`;
const LAYOUT = `${TMP}/layout`;
const FONT = "Malgun Gothic";
const C = { ink: "#000000", panel: "#EDEDED", rule: "#B8BCC4", blue: "#3D8DFF", pale: "#D0EDFA", white: "#FFFFFF" };

async function blob(pathname) {
  const b = await fs.readFile(pathname);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

function textbox(slide, text, pos, size = 24, opts = {}) {
  const s = slide.shapes.add({
    geometry: "textbox",
    position: pos,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  s.text = text;
  s.text.style = {
    typeface: FONT,
    fontSize: size,
    color: opts.color || C.ink,
    bold: opts.bold || false,
    alignment: opts.alignment || "left",
    verticalAlignment: opts.verticalAlignment || "top",
  };
  return s;
}

function rect(slide, pos, fill = C.panel, radius = false) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    position: pos,
    fill,
    line: { style: "solid", fill: fill === C.white ? C.rule : fill, width: fill === C.white ? 1 : 0 },
  });
}

function title(slide, text) {
  textbox(slide, text, { left: 42, top: 34, width: 1196, height: 74 }, 46, { bold: true });
}

function footer(slide, n, source = "형사 판결문 문체 통시 분석") {
  textbox(slide, source, { left: 42, top: 676, width: 500, height: 22 }, 14, { color: "#666666" });
  textbox(slide, String(n), { left: 1184, top: 676, width: 54, height: 22 }, 14, { alignment: "right", color: "#666666" });
}

async function addImage(slide, file, pos, fit = "contain", alt = "") {
  slide.images.add({
    blob: await blob(file),
    contentType: "image/png",
    alt,
    fit,
    position: pos,
  });
}

function bullet(slide, y, text) {
  slide.shapes.add({
    geometry: "ellipse",
    position: { left: 790, top: y + 7, width: 14, height: 14 },
    fill: C.blue,
    line: { style: "solid", fill: C.blue, width: 0 },
  });
  textbox(slide, text, { left: 824, top: y, width: 390, height: 52 }, 24);
}

async function main() {
  await fs.mkdir(PREVIEW, { recursive: true });
  await fs.mkdir(LAYOUT, { recursive: true });
  await fs.mkdir(path.dirname(OUT), { recursive: true });
  const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });

  // 1. Cover — Codex Grid slide 08 silhouette.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    textbox(s, "판결문의 말은\n어떻게 변해왔는가", { left: 42, top: 150, width: 560, height: 190 }, 70, { bold: true, verticalAlignment: "bottom" });
    textbox(s, "형사 판결문 이유부의 문체 변화 · 1980–2025", { left: 42, top: 385, width: 560, height: 70 }, 28);
    textbox(s, "명사·동사·조사 비율 · 문장 길이 · 표현 변이", { left: 42, top: 475, width: 560, height: 70 }, 21, { color: "#555555" });
    rect(s, { left: 650, top: 40, width: 588, height: 592 }, "#EAF5FB", true);
    await addImage(s, `${ROOT}/results/figures/F4_metric_heatmap.png`, { left: 670, top: 110, width: 548, height: 450 }, "contain", "문체 지표 변화 시점 히트맵");
    footer(s, 1);
  }

  // 2. Thesis.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    textbox(s, "분석이 보여주는 한 문장", { left: 42, top: 42, width: 700, height: 58 }, 26);
    textbox(s, "문장은 절반으로 짧아졌지만,\n표현은 더 명사 중심이 되었다.", { left: 42, top: 180, width: 1110, height: 205 }, 64, { bold: true, verticalAlignment: "bottom" });
    rect(s, { left: 42, top: 505, width: 1196, height: 4 }, C.blue);
    textbox(s, "짧은 문장  +  높은 명사성", { left: 42, top: 540, width: 850, height: 70 }, 36, { color: C.blue, bold: true });
    footer(s, 2);
  }

  // 3. Sample composition.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    await addImage(s, `${ROOT}/results/figures/F1_composition.png`, { left: 20, top: 20, width: 930, height: 650 }, "contain", "분석 표본 구성과 길이 변화");
    rect(s, { left: 975, top: 70, width: 250, height: 155 }, C.panel);
    textbox(s, "20,984", { left: 1000, top: 92, width: 205, height: 64 }, 44, { bold: true });
    textbox(s, "원자료 판례", { left: 1000, top: 165, width: 205, height: 35 }, 21);
    rect(s, { left: 975, top: 265, width: 250, height: 155 }, C.panel);
    textbox(s, "20,919", { left: 1000, top: 287, width: 205, height: 64 }, 44, { bold: true });
    textbox(s, "이유부 추출", { left: 1000, top: 360, width: 205, height: 35 }, 21);
    rect(s, { left: 975, top: 460, width: 250, height: 155 }, C.panel);
    textbox(s, "18,778", { left: 1000, top: 482, width: 205, height: 64 }, 44, { bold: true, color: C.blue });
    textbox(s, "본 분석 표본", { left: 1000, top: 555, width: 205, height: 35 }, 21);
    footer(s, 3, "1950–1979 보조 탐색 · 2026 부분 연도 제외");
  }

  // 4. Method timeline.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    title(s, "구성 변화와 문체 변화를 분리했다");
    textbox(s, "판결문 1건을 관측 단위로 삼고, 같은 분석기를 모든 연도에 적용했다.", { left: 42, top: 120, width: 1100, height: 52 }, 24);
    const xs = [85, 380, 675, 970];
    s.shapes.add({ geometry: "line", position: { left: 90, top: 330, width: 910, height: 0 }, line: { style: "solid", fill: C.rule, width: 2 } });
    const items = [
      ["01", "이유부 추출", "판결문 구조를 정리하고\n이유 부분만 분석"],
      ["02", "형태소 계산", "명사·동사·조사·서술어와\n문장 길이를 문서별 계산"],
      ["03", "구성 표준화", "심급 × 범죄군 분포를\n전체 기간의 고정 비중으로 보정"],
      ["04", "통계 검정", "법원 군집 표준오차와\nBH 다중검정 적용"],
    ];
    items.forEach((it, i) => {
      s.shapes.add({ geometry: "ellipse", position: { left: xs[i], top: 315, width: 30, height: 30 }, fill: i === 3 ? C.blue : C.ink, line: { style: "solid", fill: C.ink, width: 0 } });
      textbox(s, it[0], { left: xs[i], top: 245, width: 80, height: 32 }, 20, { bold: true, color: C.blue });
      textbox(s, it[1], { left: xs[i], top: 375, width: 235, height: 42 }, 28, { bold: true });
      textbox(s, it[2], { left: xs[i], top: 435, width: 245, height: 90 }, 20, { color: "#444444" });
    });
    textbox(s, "주 분석 기준: 1980–2025 · 이유부 · 형태소 150개 이상", { left: 42, top: 590, width: 900, height: 40 }, 24, { bold: true });
    footer(s, 4);
  }

  // 5. Main trends.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    await addImage(s, `${ROOT}/results/figures/F2_main_trends.png`, { left: 10, top: 5, width: 1260, height: 680 }, "contain", "원자료 추세와 사건 구성 보정 추세");
    footer(s, 5, "빈 원: 표준화 공통 셀 커버리지 80% 미만");
  }

  // 6. Effect sizes.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    await addImage(s, `${ROOT}/results/figures/F6_effect_forest.png`, { left: 18, top: 25, width: 880, height: 640 }, "contain", "보정 모형의 10년당 변화량");
    textbox(s, "해석의 핵심", { left: 930, top: 95, width: 290, height: 45 }, 28, { bold: true });
    bullet(s, 175, "명사성 증가");
    bullet(s, 270, "동사·서술어 감소");
    bullet(s, 365, "문장 길이 감소");
    textbox(s, "8개 지표 모두\nBH 보정 후 유의", { left: 930, top: 500, width: 280, height: 95 }, 28, { bold: true, color: C.blue });
    footer(s, 6, "효과 크기: 문서 간 표준편차(SD) 단위");
  }

  // 7. Variant transitions.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    await addImage(s, `${ROOT}/results/figures/F5_variant_transitions.png`, { left: 18, top: 15, width: 980, height: 660 }, "contain", "표현 변이쌍 교체 추세");
    textbox(s, "약 1991년", { left: 1020, top: 155, width: 220, height: 50 }, 32, { bold: true, color: C.blue });
    textbox(s, "아니하- → 않-", { left: 1020, top: 215, width: 220, height: 70 }, 22);
    textbox(s, "약 1995년", { left: 1020, top: 350, width: 220, height: 50 }, 32, { bold: true, color: C.blue });
    textbox(s, "하지 아니하-\n→ 하지 않-", { left: 1020, top: 410, width: 220, height: 90 }, 22);
    textbox(s, "법률문에서는 축약형보다\n기존형이 여전히 우세한 쌍도 많다.", { left: 1020, top: 550, width: 225, height: 90 }, 18, { color: "#555555" });
    footer(s, 7);
  }

  // 8. Matched examples.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    title(s, "같은 조건에서도 문장은 짧아졌다");
    textbox(s, "3심 · 재산범죄 · 연도 중앙값에 가까운 판결문", { left: 42, top: 112, width: 900, height: 40 }, 22, { color: "#555555" });
    rect(s, { left: 42, top: 195, width: 560, height: 390 }, C.panel);
    textbox(s, "1985", { left: 75, top: 225, width: 170, height: 70 }, 52, { bold: true });
    textbox(s, "명사 41.1%\n동사 4.4%\n조사 18.2%", { left: 75, top: 330, width: 220, height: 155 }, 28);
    textbox(s, "문장당 49.0어절", { left: 310, top: 360, width: 250, height: 75 }, 32, { bold: true });
    rect(s, { left: 638, top: 195, width: 600, height: 390 }, "#EAF5FB");
    textbox(s, "2025", { left: 675, top: 225, width: 170, height: 70 }, 52, { bold: true, color: C.blue });
    textbox(s, "명사 44.3%\n동사 4.1%\n조사 18.5%", { left: 675, top: 330, width: 220, height: 155 }, 28);
    textbox(s, "문장당 27.5어절", { left: 925, top: 360, width: 270, height: 75 }, 32, { bold: true, color: C.blue });
    textbox(s, "예시는 방향을 이해하기 위한 사례이며 전체 판결문을 대표하지 않는다.", { left: 42, top: 620, width: 1000, height: 34 }, 18, { color: "#666666" });
    footer(s, 8);
  }

  // 9. Limitations.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    title(s, "이 변화는 ‘사법부 전체’의 인과 효과가 아니다");
    textbox(s, "관찰된 것은 공개 판결문 코퍼스의 변화다.", { left: 42, top: 135, width: 650, height: 70 }, 30, { bold: true });
    textbox(s, "해석할 때 함께 기억할 점", { left: 790, top: 135, width: 410, height: 50 }, 26, { bold: true });
    const limits = [
      "공개 대상·수집 방식의 선택 편향",
      "오래된 문서의 구두점·입력 관행 차이",
      "현대 한국어 형태소 분석기의 역사 표기 한계",
      "표현 변이쌍은 문맥상 완전한 유의어가 아님",
    ];
    limits.forEach((t, i) => bullet(s, 225 + i * 88, t));
    textbox(s, "따라서 결론은\n‘판결문 코퍼스에서 확인된 장기 경향’으로 제한한다.", { left: 42, top: 300, width: 620, height: 150 }, 34, { bold: true, color: C.blue });
    footer(s, 9);
  }

  // 10. Close.
  {
    const s = deck.slides.add();
    s.background.fill = C.white;
    textbox(s, "결론", { left: 42, top: 42, width: 180, height: 55 }, 28);
    textbox(s, "최근 판결문은\n더 짧게 끊고,\n더 명사 중심으로 쓴다.", { left: 42, top: 155, width: 1030, height: 305 }, 76, { bold: true, verticalAlignment: "bottom" });
    rect(s, { left: 42, top: 515, width: 1196, height: 4 }, C.blue);
    textbox(s, "문장 길이의 변화와 어휘·문법 구성의 변화가 동시에 진행됐다.", { left: 42, top: 555, width: 1050, height: 58 }, 28);
    footer(s, 10);
  }

  for (const [i, slide] of deck.slides.items.entries()) {
    const stem = `slide-${String(i + 1).padStart(2, "0")}`;
    const png = await deck.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(`${PREVIEW}/${stem}.png`, new Uint8Array(await png.arrayBuffer()));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(`${LAYOUT}/${stem}.json`, await layout.text());
  }
  const montage = await deck.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(`${TMP}/deck-montage.webp`, new Uint8Array(await montage.arrayBuffer()));
  const pptx = await PresentationFile.exportPptx(deck);
  await pptx.save(OUT);
  console.log(OUT);
}

main().catch(err => {
  console.error(err);
  process.exitCode = 1;
});
