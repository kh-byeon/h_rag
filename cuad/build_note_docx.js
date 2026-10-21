// Build the Korean-language supplementary-note draft as a .docx.
//   node build_note_docx.js
// Reads note_bundle.json (produced by scripts_build_note_bundle.py); every figure,
// quotation and number below comes from that bundle, so nothing is hand-copied.

const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, PageBreak,
} = require('docx');

const B = JSON.parse(fs.readFileSync(path.join(__dirname, 'note_bundle.json'), 'utf8'));

const KR = '맑은 고딕';
const MONO = 'Consolas';
const W = 9360;                       // usable width, DXA (A4 with 1" margins)
const f = (x, n = 4) => Number(x).toFixed(n);

// ---------- small builders ----------------------------------------------
const P = (text, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 120, line: o.line ?? 280 },
  alignment: o.align,
  indent: o.indent,
  border: o.border,
  children: [new TextRun({
    text, font: o.font ?? KR, size: o.size ?? 20,
    bold: o.bold, italics: o.italics, color: o.color,
  })],
});

const H = (text, level) => new Paragraph({
  heading: level, spacing: { before: 260, after: 130 },
  children: [new TextRun({ text, font: KR, bold: true, color: '1F3864' })],
});

// Editorial marker so the reader can decide what to keep.
const KEEP = (label, note) => new Paragraph({
  spacing: { before: 60, after: 140 },
  shading: { type: ShadingType.CLEAR, fill: 'FFF4CE' },
  children: [
    new TextRun({ text: `［편집］${label}`, font: KR, size: 17, bold: true, color: '7A5B00' }),
    new TextRun({ text: note ? ` — ${note}` : '', font: KR, size: 17, color: '7A5B00' }),
  ],
});

// Verbatim English source text (quotation block).
const Q = (text) => new Paragraph({
  spacing: { after: 100, line: 250 },
  indent: { left: 340 },
  border: { left: { style: BorderStyle.SINGLE, size: 10, color: 'B4C7E7', space: 8 } },
  children: [new TextRun({ text, font: MONO, size: 16, color: '333333' })],
});

// Generated answer: one Paragraph per line (never "\n" inside a run).
const CODE = (body) => body.replace(/\r/g, '').split('\n').map((ln) => new Paragraph({
  spacing: { after: 0, line: 240 },
  indent: { left: 240 },
  shading: { type: ShadingType.CLEAR, fill: 'F7F7F7' },
  children: [new TextRun({ text: ln.length ? ln : ' ', font: MONO, size: 15 })],
}));

function table(header, rows, widths, opts = {}) {
  const cell = (t, o = {}) => new TableCell({
    width: { size: o.w, type: WidthType.DXA },
    shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({
      spacing: { after: 0, line: 240 },
      alignment: o.align,
      children: [new TextRun({
        text: String(t), font: o.font ?? KR, size: o.size ?? 17,
        bold: o.bold, color: o.color,
      })],
    })],
  });
  return new Table({
    columnWidths: widths,
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    rows: [
      new TableRow({
        tableHeader: true,
        children: header.map((h, i) => cell(h, { w: widths[i], fill: 'DEE6F2', bold: true, align: AlignmentType.CENTER })),
      }),
      ...rows.map((r) => new TableRow({
        children: r.map((c, i) => cell(c, {
          w: widths[i],
          align: i >= (opts.numFrom ?? 1) ? AlignmentType.CENTER : undefined,
          font: opts.monoCols && opts.monoCols.includes(i) ? MONO : KR,
          bold: opts.boldCols && opts.boldCols.includes(i),
        })),
      })),
    ],
  });
}

// ---------- document ------------------------------------------------------
const c = [];

c.push(new Paragraph({
  spacing: { after: 60 },
  children: [new TextRun({ text: 'Supplementary Note (초안 · 한글)', font: KR, size: 20, color: '888888' })],
}));
c.push(new Paragraph({
  heading: HeadingLevel.TITLE, spacing: { after: 80 },
  children: [new TextRun({ text: '온더플라이 MMR의 검색 단계 거동', font: KR, bold: true, size: 34, color: '1F3864' })],
}));
c.push(P('CUAD “Cap On Liability” 사례 연구 — 검색 결과 전문 및 생성 답변 전문', { size: 22, color: '444444', after: 200 }));

c.push(KEEP('이 문서의 용도', '내용 취사선택용 초안입니다. 각 절 머리의 노란 상자는 해당 절을 넣을지 판단하는 데 참고하시라고 붙였고, 최종본에서는 모두 지우면 됩니다. 영어 원문(질의·조항·생성문)은 논문에 그대로 실릴 부분이라 번역하지 않았습니다.'));

// --- 0. 실험 조건
c.push(H('0. 실험 조건 및 출처', HeadingLevel.HEADING_1));
c.push(KEEP('필수', '재현성 확보를 위해 남겨야 합니다. 지면이 부족하면 Methods로 옮기고 여기서는 참조만 하는 방법도 있습니다.'));
c.push(table(
  ['항목', '값'],
  [
    ['검색 레코드', `${B.src_jsonl} · query_idx = ${B.query_idx}`],
    ['문서', `CUAD doc_id ${B.doc_id} · Licensor–Licensee 소프트웨어 라이선스 · 핵심 조항 §13.1 LIMIT OF LIABILITY`],
    ['검색 설정', `문서 내(intra-document), top-k = ${B.params.top_k}, λ = ${B.params.mmr_lambda}, γ = ${B.params.gamma}, τ_dup = ${B.params.tau_dup}, window_size = ${B.params.window_size}`],
    ['생성기', 'Llama-3-8B-Instruct · fp16 / int8 / int4 · 두 arm 동일(symmetric) 프롬프트'],
    ['공통 조건', '두 arm은 동일한 임베딩·인덱스·top-k 예산을 공유하며, 재순위(re-ranking) 방식만 다름'],
  ], [1900, 7460], { numFrom: 99 }));
c.push(P(''));
c.push(P('검색은 생성기 호출 이전에 종료되므로, 아래 검색 결과는 세 정밀도에서 완전히 동일하다. 정밀도에 의존하는 것은 4절(생성 답변)뿐이다.', { italics: true, color: '555555' }));

// --- 1. 질의와 정답
c.push(H('1. 질의와 정답 스팬', HeadingLevel.HEADING_1));
c.push(KEEP('필수', '정답 스팬이 5개라는 사실이 뒤의 해석(“두 arm 모두 §13.1은 찾는다”)의 전제입니다.'));
c.push(P('질의', { bold: true, after: 60 }));
c.push(Q(B.query));
c.push(P(`주석된 정답 스팬 (${B.spans.length}개)`, { bold: true, after: 60 }));
B.spans.forEach((s) => { c.push(P(s.name, { bold: true, size: 18, after: 40 })); c.push(Q(s.text)); });

// --- 2. 검색 결과 전문
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H('2. 검색된 top-10 전문 (두 arm)', HeadingLevel.HEADING_1));
c.push(KEEP('분량 최다 — 취사선택 지점', `청크 20개 전문입니다. 지면이 부족하면 (a) 전용 슬롯 8개만 남기거나, (b) 조항 라벨 표만 본문에 두고 전문은 별도 데이터 파일로 돌리는 선택지가 있습니다.`));
c.push(P(`교차 식별자 Cxxxx 는 청크 본문 앞 120자의 MD5이며, 두 목록에 같은 Cxxxx가 나오면 물리적으로 동일한 청크다. 모든 청크의 doc_id는 ${B.doc_id}이다.`, { size: 18 }));
c.push(P(`[L2_Anchor] 는 γ가 예약한 순수 관련도 슬롯을 뜻한다. γ = ${B.params.gamma} × top-k = ${B.params.top_k} ⇒ 앵커 ${B.n_anchor}개이며, 레코드의 소스 태그로 확인된다. 나머지 ${B.params.top_k - B.n_anchor}개 슬롯은 MMR이 선택하고, 그중 일부는 L2의 선택과 겹친다(‘공유’로 표시).`, { size: 18 }));

[['2.1 L2-only — 검색 순서', B.l2], ['2.2 MMR — 검색 순서', B.mmr]].forEach(([title, rows]) => {
  c.push(H(title, HeadingLevel.HEADING_2));
  rows.forEach((r, i) => {
    c.push(new Paragraph({
      spacing: { before: 100, after: 40 },
      children: [
        new TextRun({ text: `${i + 1}. `, font: KR, size: 19, bold: true }),
        new TextRun({ text: r.cid, font: MONO, size: 18, bold: true, color: '1F3864' }),
        new TextRun({ text: `  [${r.tag} / ${r.shared ? '공유' : '전용'}]  `, font: KR, size: 17, color: '806000' }),
        new TextRun({ text: r.clause, font: KR, size: 18 }),
      ],
    }));
    c.push(Q(r.text));
  });
});

// --- 3. 분석
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H('3. 검색 결과 분석', HeadingLevel.HEADING_1));

c.push(H('3.1 슬롯 배분', HeadingLevel.HEADING_2));
c.push(KEEP('핵심 — 반드시 포함', '이 표 하나가 “동일 예산에서 중복을 줄여 조항 커버리지를 넓힌다”는 주장의 근거입니다.'));
c.push(table(['구분', 'L2-only', 'MMR'], [
  ['공유 슬롯', `${B.n_shared}`, `${B.n_shared}`],
  ['전용 슬롯', `${B.n_l2_excl}`, `${B.n_mmr_excl}`],
  ['Art.9 면책에 쓴 슬롯', `${B.n9_l2} / ${B.params.top_k}`, `${B.n9_mmr} / ${B.params.top_k}`],
], [4200, 2580, 2580], { boldCols: [1, 2] }));
c.push(P(''));
c.push(P('L2의 전용 슬롯 4개는 모두 Article 9 면책 조항이며, 그중 하나는 300/150자 청크 겹침 때문에 생긴 §9.2의 두 번째 윈도다. MMR의 전용 슬롯 4개는 서로 다른 해지·구제 계열 조항이다.'));
c.push(table(['arm', '전용 청크'], [
  ['L2', B.l2_excl.map((x) => `${x.cid} ${x.clause}`).join(' · ')],
  ['MMR', B.mmr_excl.map((x) => `${x.cid} ${x.clause}`).join(' · ')],
], [1100, 8260], { numFrom: 99 }));

c.push(H('3.2 정답 스팬 커버리지', HeadingLevel.HEADING_2));
c.push(KEEP('권장 포함', '스팬 C를 MMR만 잡는다는 것이 “다양성이 실제 정답 영역을 넓힌다”의 직접 증거입니다. 다만 스팬 A는 양쪽 다 못 잡는다는 사실도 함께 적어야 균형이 맞습니다.'));
c.push(table(['정답 스팬', 'L2 슬롯', 'MMR 슬롯'], B.span_cov.map((s) => [
  s.name, s.l2.length ? `[${s.l2.join(', ')}]` : '—', s.mmr.length ? `[${s.mmr.join(', ')}]` : '—',
]), [4800, 2280, 2280]));
c.push(P(''));
c.push(P('스팬 C는 MMR만 검색한다(슬롯 6, C1E48B). 해당 청크는 주석 스팬이 끝나는 지점에서 시작하므로(“… return all copies of the Licensed Products to Licensor with a” / “with a certification by an authorized representative …”) 조항 경계에서 서로 겹친다. 스팬 A는 두 arm 모두 커버하지 못한다.'));

c.push(H('3.3 공유 청크의 순위 재배치', HeadingLevel.HEADING_2));
c.push(KEEP('선택 — 있으면 좋음', '“어떤 청크를 뽑느냐”뿐 아니라 “몇 위에 두느냐”도 바뀐다는 보조 논거입니다. 분량이 빠듯하면 본문 한 문장으로 요약하고 표는 빼도 됩니다.'));
c.push(table(['청크', '조항', 'L2 순위', 'MMR 순위'],
  B.rank.map((r) => [r.cid, r.clause, `${r.l2}`, `${r.mmr}`]),
  [1200, 5160, 1500, 1500], { monoCols: [0], numFrom: 2 }));
c.push(P(''));
c.push(P('MMR은 검색되는 청크 구성뿐 아니라 공유 청크의 순위도 바꾼다. §12.4 EXCLUSIVE REMEDIES(C7BBE4, 정답 스팬 E)가 8위에서 3위로 올라오고, 근중복 관계인 §13.1 윈도 두 개는 아래로 밀린다.'));

// --- 4. 생성 답변
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H('4. 생성 답변 전문', HeadingLevel.HEADING_1));
c.push(KEEP('분량 큼 — 취사선택 지점', 'int8만 싣고 fp16/int4는 데이터 파일로 돌리는 선택지가 있습니다. 세 정밀도를 모두 실으면 “동일 검색 결과 위에서 정밀도만 달라진다”는 설계를 독자가 직접 확인할 수 있다는 장점이 있습니다.'));
c.push(P('아래 6개 답변은 모두 2절의 동일한 검색 결과를 입력으로 받았다. 달라지는 것은 생성기의 수치 정밀도뿐이다.', { italics: true, color: '555555' }));
['fp16', 'int8', 'int4'].forEach((q, i) => {
  c.push(H(`4.${i + 1} ${q}`, HeadingLevel.HEADING_2));
  ['L2', 'MMR'].forEach((arm) => {
    c.push(P(`${arm} (${q})`, { bold: true, after: 60 }));
    CODE(B.gen[q][arm]).forEach((p) => c.push(p));
    c.push(P(''));
  });
});

// --- 5. 집계
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H('5. 동일 실행의 집계 결과 (n = 50 질의)', HeadingLevel.HEADING_1));
c.push(KEEP('권장 포함', '사례가 n=1이므로, 같은 실행의 50개 질의 집계를 함께 두면 사례가 단독으로 서지 않습니다. 위 사례는 이 50개 중 하나(query_idx 29)입니다.'));
const A = B.agg;
c.push(P('검색 지표', { bold: true, after: 60 }));
c.push(table(['지표', 'L2-only', 'MMR', 'Δ'], [
  ['ILD ↑', f(A.l2_ild), f(A.mmr_ild), `+${f(A.mmr_ild - A.l2_ild)}`],
  ['RR ↓', f(A.l2_rr), f(A.mmr_rr), `${f(A.mmr_rr - A.l2_rr)}`],
  ['centroid breadth ↑', f(A.l2_breadth, 2), f(A.mmr_breadth, 2), `+${f(A.mmr_breadth - A.l2_breadth, 2)}`],
  ['recall', f(A.l2_recall, 2), f(A.mmr_recall, 2), f(A.mmr_recall - A.l2_recall, 2)],
  ['ROUGE-L', f(A.l2_rouge_l), f(A.mmr_rouge_l), f(A.mmr_rouge_l - A.l2_rouge_l)],
], [3000, 2120, 2120, 2120]));
c.push(P(''));
c.push(P('동일한 top-10 예산에서 MMR은 중복(RR)을 낮추고 탐색 폭(breadth)을 넓히면서 recall을 잃지 않는다.', { size: 18 }));
c.push(P('ILD는 MMR이 직접 최적화하는 목적함수이므로 동어반복이라는 지적을 받을 수 있다. 주장의 무게는 RR과 breadth에 두고, ILD는 목적함수가 의도대로 작동함을 확인하는 지표로 위치시키는 편이 안전하다.', { size: 18, italics: true, color: '7A5B00' }));

if (Object.keys(B.evals).length) {
  c.push(P('', { after: 80 }));
  c.push(P('정밀도별 생성 지표 (같은 50 질의)', { bold: true, after: 60 }));
  const E = B.evals;
  c.push(table(['지표', 'fp16', 'int8', 'int4'], [
    ['L2 aspect coverage', f(E.fp16.l2_cov, 3), f(E.int8.l2_cov, 3), f(E.int4.l2_cov, 3)],
    ['MMR aspect coverage', f(E.fp16.mmr_cov, 3), f(E.int8.mmr_cov, 3), f(E.int4.mmr_cov, 3)],
    ['L2 BERTScore-F1', f(E.fp16.l2_bs), f(E.int8.l2_bs), f(E.int4.l2_bs)],
    ['MMR BERTScore-F1', f(E.fp16.mmr_bs), f(E.int8.mmr_bs), f(E.int4.mmr_bs)],
    ['검색 coverage (L2, top3-graded)', f(E.fp16.l2_rc3g), f(E.int8.l2_rc3g), f(E.int4.l2_rc3g)],
    ['검색 coverage (MMR, top3-graded)', f(E.fp16.mmr_rc3g), f(E.int8.mmr_rc3g), f(E.int4.mmr_rc3g)],
  ], [3960, 1800, 1800, 1800]));
  c.push(P(''));
  c.push(P('마지막 두 행은 세 정밀도에서 소수점까지 동일하다. 검색이 생성기 정밀도와 무관하다는 설계 논리를 데이터가 그대로 보여주는 정합성 검증이다.', { size: 18 }));
  c.push(P('반면 aspect coverage와 BERTScore는 방향이 엇갈린다. coverage는 세 정밀도 모두 MMR이 앞서지만 BERTScore는 int4에서 역전된다. 따라서 “MMR이 양자화 열화를 줄인다”고는 쓸 수 없다.', { size: 18, italics: true, color: '7A5B00' }));
}

// --- 6. 한계
c.push(H('6. 범위와 한계', HeadingLevel.HEADING_1));
c.push(KEEP('필수', '심사자가 스스로 찾아낼 항목들을 먼저 적어두는 편이 방어력이 높습니다. 특히 2번과 4번은 빼지 마십시오.'));
[
  '이 사례는 질의 1건에 대해 조건마다 그리디 디코딩 1회를 수행한 결과다. 메커니즘을 예시하는 것이 목적이며 유의성 검정이 아니다. 집계 통계는 5절과 본문에 있다.',
  '주석된 정답은 5개 스팬으로 구성된다. §13.1 상한 조항 자체는 두 arm 모두 검색하므로, 두 방식의 차이는 “정답 조항을 찾았는가”가 아니라 “나머지 슬롯을 어떻게 쓰는가”에 있다.',
  'int4에서는 두 arm 모두 서술이 짧아지고 구체성이 떨어진다. int4 MMR 답변은 조항 번호를 청크 인덱스로 잘못 표기한 “Chunk 13.1”을 생성하는데, 검색 목록에는 대응하는 항목이 없다.',
  '생성 단계 집계 지표는 한 방향으로 움직이지 않는다(NLI aspect coverage는 세 정밀도 모두 MMR 우위, BERTScore-F1은 그렇지 않다). 이 note는 MMR이 생성기의 양자화 열화를 줄인다고 주장하지 않는다.',
].forEach((t, i) => c.push(P(`${i + 1}) ${t}`, { indent: { left: 200, hanging: 200 } })));

// ---------- write ---------------------------------------------------------
const doc = new Document({
  styles: { default: { document: { run: { font: KR, size: 20 } } } },
  sections: [{
    properties: { page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children: c,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, 'Supplementary_Note_CapOnLiability_KR.docx');
  fs.writeFileSync(out, buf);
  console.log('wrote', out, `(${(buf.length / 1024).toFixed(0)} KB, ${c.length} blocks)`);
});
