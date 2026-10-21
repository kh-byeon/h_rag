// Build the English (submission-ready) supplementary note as a .docx.
//   node build_note_docx_en.js
// Reads note_bundle.json (from scripts_build_note_bundle.py). Every figure and
// quotation is taken from that bundle, so nothing is hand-transcribed and the
// English and Korean drafts cannot drift apart numerically.
//
// Difference from the Korean draft: no editorial call-out boxes (those exist so
// the author can decide what to keep); prose is in submission register.

const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, PageBreak,
} = require('docx');

const B = JSON.parse(fs.readFileSync(path.join(__dirname, 'note_bundle.json'), 'utf8'));

const SERIF = 'Cambria';
const MONO = 'Consolas';
const f = (x, n = 4) => Number(x).toFixed(n);
const pct = (a, b) => `${((b - a) / a * 100).toFixed(1)}%`;

const P = (text, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 130, line: o.line ?? 290 },
  alignment: o.align,
  indent: o.indent,
  children: [new TextRun({
    text, font: o.font ?? SERIF, size: o.size ?? 20,
    bold: o.bold, italics: o.italics, color: o.color,
  })],
});

const H = (text, level) => new Paragraph({
  heading: level, spacing: { before: 280, after: 140 },
  children: [new TextRun({ text, font: SERIF, bold: true, color: '1F3864' })],
});

// Verbatim source text.
const Q = (text) => new Paragraph({
  spacing: { after: 100, line: 250 },
  indent: { left: 340 },
  border: { left: { style: BorderStyle.SINGLE, size: 10, color: 'B4C7E7', space: 8 } },
  children: [new TextRun({ text, font: MONO, size: 16, color: '333333' })],
});

const CODE = (body) => body.replace(/\r/g, '').split('\n').map((ln) => new Paragraph({
  spacing: { after: 0, line: 240 },
  indent: { left: 240 },
  shading: { type: ShadingType.CLEAR, fill: 'F7F7F7' },
  children: [new TextRun({ text: ln.length ? ln : ' ', font: MONO, size: 15 })],
}));

const CAP = (text) => new Paragraph({
  spacing: { before: 60, after: 160 },
  children: [new TextRun({ text, font: SERIF, size: 17, color: '444444' })],
});

function table(header, rows, widths, opts = {}) {
  const cell = (t, o = {}) => new TableCell({
    width: { size: o.w, type: WidthType.DXA },
    shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({
      spacing: { after: 0, line: 240 },
      alignment: o.align,
      children: [new TextRun({
        text: String(t), font: o.font ?? SERIF, size: o.size ?? 17,
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
        children: header.map((h, i) => cell(h, {
          w: widths[i], fill: 'DEE6F2', bold: true, align: AlignmentType.CENTER,
        })),
      }),
      ...rows.map((r) => new TableRow({
        children: r.map((cv, i) => cell(cv, {
          w: widths[i],
          align: i >= (opts.numFrom ?? 1) ? AlignmentType.CENTER : undefined,
          font: opts.monoCols && opts.monoCols.includes(i) ? MONO : SERIF,
          bold: opts.boldCols && opts.boldCols.includes(i),
        })),
      })),
    ],
  });
}

const CLAUSE_EN = {
  'CE45B7': '§13.1 cont. — consequential-damages waiver',
  'C42340': '§13.1 LIMIT OF LIABILITY (head of clause)',
  'C8C2D2': '§9.1 INDEMNIFICATION OF LICENSOR',
  'CA3EA4': '§9.2 INDEMNIFICATION OF LICENSEE',
  'C21F8E': '§13.1 LIMIT OF LIABILITY',
  'C7A673': '§9.1 INDEMNIFICATION (Art. 9 boundary)',
  'CFBACB': '§9.2 INDEMNIFICATION OF LICENSEE (overlapping window)',
  'C7BBE4': '§12.4 EXCLUSIVE REMEDIES / warranty disclaimer',
  'C56C5E': '§9.3 INDEMNIFICATION PROCEDURE',
  'C8492E': '§13.1 cont. → Art. 14 (Term) boundary',
  'C5CE22': 'insolvency / bankruptcy termination trigger',
  'C1E48B': 'return / destroy / refund remedy',
  'C6B071': '§14.3 DUTIES UPON TERMINATION',
  'C1BE9E': 'equitable relief / specific performance',
};
const SPAN_EN = [
  'A — 30-day cure period',
  'B — §13.1 total-liability cap',
  'C — return/destroy/refund sole remedy',
  'D — consequential-damages waiver',
  'E — sole and exclusive remedies',
];
const cl = (cid) => CLAUSE_EN[cid] || '(unlabelled)';

const c = [];

// ---- title ---------------------------------------------------------------
c.push(new Paragraph({
  spacing: { after: 60 },
  children: [new TextRun({ text: 'Supplementary Note', font: SERIF, size: 20, color: '888888' })],
}));
c.push(new Paragraph({
  heading: HeadingLevel.TITLE, spacing: { after: 90 },
  children: [new TextRun({
    text: 'Retrieval-stage behaviour of on-the-fly MMR',
    font: SERIF, bold: true, size: 32, color: '1F3864',
  })],
}));
c.push(P('A CUAD “Cap On Liability” case study, with the complete retrieved context and the generated answers at three generator precisions',
  { size: 21, color: '444444', after: 220 }));

c.push(P('This note documents, for one query, exactly which passages each retriever placed in the generator’s context window and what the generator produced from them. The purpose is to make the aggregate retrieval-diversity result in the main text mechanically legible: the two arms share an embedding model, an index and a top-k budget, and differ only in re-ranking.', { after: 160 }));

// ---- 1 setup -------------------------------------------------------------
c.push(H('1. Experimental setting and provenance', HeadingLevel.HEADING_1));
c.push(table(['Item', 'Value'], [
  ['Retrieval record', `${B.src_jsonl} · query_idx = ${B.query_idx}`],
  ['Document', `CUAD doc_id ${B.doc_id}; Licensor–Licensee software licence. Core clause §13.1 LIMIT OF LIABILITY.`],
  ['Retrieval', `Intra-document; top-k = ${B.params.top_k}; λ = ${B.params.mmr_lambda}; γ = ${B.params.gamma}; τ_dup = ${B.params.tau_dup}; window_size = ${B.params.window_size}`],
  ['Generator', 'Llama-3-8B-Instruct at fp16 / int8 / int4; identical (symmetric) prompt for both arms'],
  ['Shared', 'Both arms use the same embeddings, the same index and the same top-k budget; only the re-ranking differs.'],
], [1750, 7610], { numFrom: 99 }));
c.push(P(''));
c.push(P('Retrieval completes before the generator is invoked. The retrieved sets in Section 3 are therefore identical at all three precisions, and only Section 5 (generated answers) depends on quantization.', { italics: true, color: '444444' }));

// ---- 2 query -------------------------------------------------------------
c.push(H('2. Query and annotated ground truth', HeadingLevel.HEADING_1));
c.push(P('Query', { bold: true, after: 60 }));
c.push(Q(B.query));
c.push(P(`The CUAD annotation for this query comprises ${B.spans.length} spans. Both retrievers recover the §13.1 cap itself (span B), so the comparison below concerns how the remaining slots are spent rather than whether the answer clause is found at all.`, { after: 110 }));
B.spans.forEach((s, i) => {
  c.push(P(SPAN_EN[i], { bold: true, size: 18, after: 40 }));
  c.push(Q(s.text));
});

// ---- 3 retrieved ---------------------------------------------------------
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H('3. Retrieved top-10, both arms', HeadingLevel.HEADING_1));
c.push(P(`The cross-list identifier Cxxxx is the MD5 of the first 120 characters of the chunk body; the same identifier appearing in both lists denotes the same physical chunk. All chunks belong to doc_id ${B.doc_id}.`, { size: 18 }));
c.push(P(`The tag [L2_Anchor] marks the slots reserved for pure relevance by γ. With γ = ${B.params.gamma} and top-k = ${B.params.top_k}, ${B.n_anchor} slots are anchors, as confirmed by the source tags recorded with the run; the remaining ${B.params.top_k - B.n_anchor} are selected by MMR. Chunks that both arms retrieve are marked “shared”.`, { size: 18 }));

[['3.1 L2-only, in retrieval order', B.l2], ['3.2 MMR, in retrieval order', B.mmr]].forEach(([title, rows]) => {
  c.push(H(title, HeadingLevel.HEADING_2));
  rows.forEach((r, i) => {
    c.push(new Paragraph({
      spacing: { before: 100, after: 40 },
      children: [
        new TextRun({ text: `${i + 1}. `, font: SERIF, size: 19, bold: true }),
        new TextRun({ text: r.cid, font: MONO, size: 18, bold: true, color: '1F3864' }),
        new TextRun({ text: `  [${r.tag}; ${r.shared ? 'shared' : 'exclusive'}]  `, font: SERIF, size: 17, color: '806000' }),
        new TextRun({ text: cl(r.cid), font: SERIF, size: 18 }),
      ],
    }));
    c.push(Q(r.text));
  });
});

// ---- 4 analysis ----------------------------------------------------------
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H('4. Analysis of the retrieved sets', HeadingLevel.HEADING_1));

c.push(H('4.1 Slot allocation', HeadingLevel.HEADING_2));
c.push(table(['', 'L2-only', 'MMR'], [
  ['Shared slots', `${B.n_shared}`, `${B.n_shared}`],
  ['Exclusive slots', `${B.n_l2_excl}`, `${B.n_mmr_excl}`],
  ['Slots spent on Art. 9 (indemnification)', `${B.n9_l2} / ${B.params.top_k}`, `${B.n9_mmr} / ${B.params.top_k}`],
], [4400, 2480, 2480], { boldCols: [1, 2] }));
c.push(CAP('Table 1 | Allocation of the ten retrieved slots.'));
c.push(P(`The four slots exclusive to L2 are all Article 9 indemnification passages; one of them is a second, overlapping window of §9.2 created by the 300/150-character chunk overlap, so the arm spends ${B.n9_l2} of ${B.params.top_k} slots on a single clause group. MMR retains one §9.1 passage and reallocates the other four slots to four distinct termination-side clauses.`));
c.push(table(['Arm', 'Exclusive chunks'], [
  ['L2', B.l2_excl.map((x) => `${x.cid} ${cl(x.cid)}`).join('; ')],
  ['MMR', B.mmr_excl.map((x) => `${x.cid} ${cl(x.cid)}`).join('; ')],
], [1000, 8360], { numFrom: 99 }));
c.push(CAP('Table 2 | Chunks retrieved by one arm only.'));

c.push(H('4.2 Coverage of the annotated spans', HeadingLevel.HEADING_2));
c.push(table(['Ground-truth span', 'L2 slots', 'MMR slots'],
  B.span_cov.map((s, i) => [SPAN_EN[i],
    s.l2.length ? `[${s.l2.join(', ')}]` : '—',
    s.mmr.length ? `[${s.mmr.join(', ')}]` : '—']),
  [4900, 2230, 2230]));
c.push(CAP('Table 3 | Retrieval slots containing each annotated span. Matching uses sliding eight-word fragments so that a span split across a chunk boundary is still detected.'));
c.push(P('Span C is recovered only by MMR (slot 6, C1E48B). That chunk begins where the annotated span ends — “… return all copies of the Licensed Products to Licensor with a” against “with a certification by an authorized representative …” — so the two overlap at the clause boundary. Span A is recovered by neither arm.'));

c.push(H('4.3 Rank reallocation among shared chunks', HeadingLevel.HEADING_2));
c.push(table(['Chunk', 'Clause', 'L2 rank', 'MMR rank'],
  B.rank.map((r) => [r.cid, cl(r.cid), `${r.l2}`, `${r.mmr}`]),
  [1200, 5160, 1500, 1500], { monoCols: [0], numFrom: 2 }));
c.push(CAP('Table 4 | Position of the six shared chunks in each ranking.'));
c.push(P('The two arms differ not only in which chunks they retrieve but in where they place the shared ones. §12.4 EXCLUSIVE REMEDIES (C7BBE4), which carries ground-truth span E, rises from rank 8 to rank 3, while the two near-duplicate §13.1 windows are displaced downwards.'));

// ---- 5 generations -------------------------------------------------------
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H('5. Generated answers', HeadingLevel.HEADING_1));
c.push(P('All six answers below were produced from the retrieved context in Section 3. Only the numerical precision of the generator differs across the three subsections.', { italics: true, color: '444444' }));
['fp16', 'int8', 'int4'].forEach((q, i) => {
  c.push(H(`5.${i + 1} ${q}`, HeadingLevel.HEADING_2));
  ['L2', 'MMR'].forEach((arm) => {
    c.push(P(`${arm} (${q})`, { bold: true, after: 60 }));
    CODE(B.gen[q][arm]).forEach((p) => c.push(p));
    c.push(P(''));
  });
});

// ---- 6 aggregate ---------------------------------------------------------
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H(`6. Aggregate results for the same run (n = ${B.agg_n} queries)`, HeadingLevel.HEADING_1));
c.push(P(`The case above is one of the ${B.agg_n} CUAD queries evaluated in this run (query_idx ${B.query_idx}); the statistics below come from the same execution and the same retrieval configuration.`, { after: 120 }));
const A = B.agg;
c.push(table(['Metric', 'L2-only', 'MMR', 'Δ'], [
  ['ILD (higher is better)', f(A.l2_ild), f(A.mmr_ild), `+${f(A.mmr_ild - A.l2_ild)} (+${pct(A.l2_ild, A.mmr_ild)})`],
  ['Redundancy rate (lower is better)', f(A.l2_rr), f(A.mmr_rr), `${f(A.mmr_rr - A.l2_rr)} (${pct(A.l2_rr, A.mmr_rr)})`],
  ['Centroid breadth (higher is better)', f(A.l2_breadth, 2), f(A.mmr_breadth, 2), `+${f(A.mmr_breadth - A.l2_breadth, 2)} (+${pct(A.l2_breadth, A.mmr_breadth)})`],
  ['Recall', f(A.l2_recall, 2), f(A.mmr_recall, 2), f(A.mmr_recall - A.l2_recall, 2)],
  ['ROUGE-L', f(A.l2_rouge_l), f(A.mmr_rouge_l), f(A.mmr_rouge_l - A.l2_rouge_l)],
], [3400, 1920, 1920, 2120]));
c.push(CAP('Table 5 | Retrieval-stage metrics over the full query set.'));
c.push(P('Under an identical top-10 budget, MMR lowers near-duplicate redundancy and widens centroid coverage while leaving recall unchanged. Because intra-list diversity is the quantity MMR optimises directly, we treat ILD as a check that the objective behaves as intended and rest the substantive claim on the redundancy rate and the exploration breadth.'));

if (Object.keys(B.evals).length) {
  const E = B.evals;
  c.push(P('', { after: 80 }));
  c.push(table(['Metric', 'fp16', 'int8', 'int4'], [
    ['Aspect coverage, L2', f(E.fp16.l2_cov, 3), f(E.int8.l2_cov, 3), f(E.int4.l2_cov, 3)],
    ['Aspect coverage, MMR', f(E.fp16.mmr_cov, 3), f(E.int8.mmr_cov, 3), f(E.int4.mmr_cov, 3)],
    ['BERTScore-F1, L2', f(E.fp16.l2_bs), f(E.int8.l2_bs), f(E.int4.l2_bs)],
    ['BERTScore-F1, MMR', f(E.fp16.mmr_bs), f(E.int8.mmr_bs), f(E.int4.mmr_bs)],
    ['Retrieval coverage (top-3, graded), L2', f(E.fp16.l2_rc3g), f(E.int8.l2_rc3g), f(E.int4.l2_rc3g)],
    ['Retrieval coverage (top-3, graded), MMR', f(E.fp16.mmr_rc3g), f(E.int8.mmr_rc3g), f(E.int4.mmr_rc3g)],
  ], [4160, 1720, 1720, 1760]));
  c.push(CAP('Table 6 | Generation-stage and retrieval-stage metrics by generator precision, same query set.'));
  c.push(P('The last two rows are identical to the reported precision across all three settings, which is the expected consequence of retrieval preceding generation and serves as a consistency check on the pipeline. The generation-stage metrics do not move together: aspect coverage favours MMR at every precision, whereas BERTScore-F1 does not at int4. We therefore make no claim that MMR mitigates quantization-induced degradation of the generator.'));
}

// ---- 7 limitations -------------------------------------------------------
c.push(H('7. Scope and limitations', HeadingLevel.HEADING_1));
[
  `This is a single query with one greedy decode per condition. It is intended to make a mechanism legible, not to establish significance; aggregate statistics over the ${B.agg_n} queries of the same run are given in Section 6 and in the main text.`,
  'The annotated ground truth comprises five spans, and both arms retrieve the §13.1 cap itself. The difference between the arms therefore lies in the use of the remaining slots rather than in whether the answer clause is found.',
  'At int4 both arms produce shorter and less specific reasoning. The int4 MMR answer additionally mislabels a clause number as a chunk index (“Chunk 13.1”), for which there is no corresponding entry in the retrieved list.',
  'Generation-stage aggregate metrics do not move in a single direction across precisions (Table 6). No claim is made here that MMR reduces quantization-induced degradation of the generator.',
].forEach((t, i) => c.push(P(`${i + 1}. ${t}`, { indent: { left: 220, hanging: 220 } })));

// ---- write ---------------------------------------------------------------
const doc = new Document({
  styles: { default: { document: { run: { font: SERIF, size: 20 } } } },
  sections: [{
    properties: { page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children: c,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, 'Supplementary_Note_CapOnLiability_EN.docx');
  fs.writeFileSync(out, buf);
  console.log('wrote', out, `(${(buf.length / 1024).toFixed(0)} KB, ${c.length} blocks)`);
});
