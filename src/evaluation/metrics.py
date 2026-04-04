from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd

HAS_SACREBLEU = False
try:
    import sacrebleu
    HAS_SACREBLEU = True
except Exception:
    sacrebleu = None

HAS_ROUGE = False
try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except Exception:
    rouge_scorer = None

HAS_BERTSCORE = False
try:
    from bert_score import score as bertscore_score
    HAS_BERTSCORE = True
except Exception:
    bertscore_score = None

HAS_SBERT = False
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SBERT = True
except Exception:
    SentenceTransformer = None
    util = None

HAS_EASSE = False
try:
    from easse.sari import corpus_sari
    HAS_EASSE = True
except Exception:
    corpus_sari = None


_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip()


def normalize_spaces(text: str) -> str:
    text = safe_text(text)
    return re.sub(r"\s+", " ", text).strip()


def simple_word_tokenize(text: str) -> list[str]:
    text = normalize_spaces(text).lower()
    return _WORD_RE.findall(text)


def split_sentences(text: str) -> list[str]:
    text = normalize_spaces(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [text]


def word_count(text: str) -> int:
    return len(simple_word_tokenize(text))


VOWELS = "aeiouáéíóúü"


def count_syllables_word_es(word: str) -> int:
    word = safe_text(word).lower()
    word = re.sub(r"[^a-záéíóúüñ]", "", word)

    if not word:
        return 0

    groups = re.findall(r"[aeiouáéíóúü]+", word)
    syllables = len(groups)

    return max(1, syllables)


def count_syllables_text_es(text: str) -> int:
    tokens = simple_word_tokenize(text)
    if not tokens:
        return 0
    return sum(count_syllables_word_es(tok) for tok in tokens)


def fernandez_huerta(text: str) -> float | None:
    text = normalize_spaces(text)
    words = word_count(text)
    sents = len(split_sentences(text))
    sylls = count_syllables_text_es(text)

    if words == 0 or sents == 0:
        return None

    P = (sylls / words) * 100.0
    F = (sents / words) * 100.0

    return 206.84 - (0.60 * P) - (1.02 * F)


def inflesz(text: str) -> float | None:
    text = normalize_spaces(text)
    words = word_count(text)
    sents = len(split_sentences(text))
    sylls = count_syllables_text_es(text)

    if words == 0 or sents == 0:
        return None

    return 206.835 - 62.3 * (sylls / words) - (words / sents)


def compression_ratio(source: str, prediction: str) -> float | None:
    src_words = word_count(source)
    pred_words = word_count(prediction)
    if src_words == 0:
        return None
    return pred_words / src_words


def sentence_splits(source: str, prediction: str) -> int | None:
    src_s = len(split_sentences(source))
    pred_s = len(split_sentences(prediction))
    if src_s == 0:
        return None
    return pred_s - src_s


def levenshtein_similarity(source: str, prediction: str) -> float:
    source = normalize_spaces(source)
    prediction = normalize_spaces(prediction)
    if not source and not prediction:
        return 1.0
    return SequenceMatcher(None, source, prediction).ratio()


def exact_copy(source: str, prediction: str) -> int:
    return int(normalize_spaces(source) == normalize_spaces(prediction))


def additions_proportion(source: str, prediction: str) -> float | None:
    src_tokens = simple_word_tokenize(source)
    pred_tokens = simple_word_tokenize(prediction)

    if len(pred_tokens) == 0:
        return 0.0

    src_counter = Counter(src_tokens)
    pred_counter = Counter(pred_tokens)

    added = 0
    for tok, c_pred in pred_counter.items():
        c_src = src_counter.get(tok, 0)
        if c_pred > c_src:
            added += (c_pred - c_src)

    return added / len(pred_tokens)


def deletions_proportion(source: str, prediction: str) -> float | None:
    src_tokens = simple_word_tokenize(source)
    pred_tokens = simple_word_tokenize(prediction)

    if len(src_tokens) == 0:
        return None

    src_counter = Counter(src_tokens)
    pred_counter = Counter(pred_tokens)

    deleted = 0
    for tok, c_src in src_counter.items():
        c_pred = pred_counter.get(tok, 0)
        if c_src > c_pred:
            deleted += (c_src - c_pred)

    return deleted / len(src_tokens)


def compute_bleu(reference: str, prediction: str) -> float | None:
    reference = normalize_spaces(reference)
    prediction = normalize_spaces(prediction)

    if not reference or not prediction:
        return None

    if HAS_SACREBLEU:
        try:
            score = sacrebleu.corpus_bleu([prediction], [[reference]])
            return float(score.score)
        except Exception:
            return None

    return None


def compute_rouge(reference: str, prediction: str) -> dict[str, float | None]:
    reference = normalize_spaces(reference)
    prediction = normalize_spaces(prediction)

    out = {
        "rouge1_f": None,
        "rouge2_f": None,
        "rougeL_f": None,
    }

    if not reference or not prediction or not HAS_ROUGE:
        return out

    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        scores = scorer.score(reference, prediction)
        out["rouge1_f"] = float(scores["rouge1"].fmeasure)
        out["rouge2_f"] = float(scores["rouge2"].fmeasure)
        out["rougeL_f"] = float(scores["rougeL"].fmeasure)
    except Exception:
        pass

    return out


def _ngrams(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def _sari_fallback_single(source: str, prediction: str, reference: str) -> float | None:
    source_tokens = simple_word_tokenize(source)
    pred_tokens = simple_word_tokenize(prediction)
    ref_tokens = simple_word_tokenize(reference)

    if not source_tokens or not prediction or not reference:
        return None

    add_scores = []
    keep_scores = []
    del_scores = []

    for n in range(1, 5):
        s_ngr = set(_ngrams(source_tokens, n).keys())
        p_ngr = set(_ngrams(pred_tokens, n).keys())
        r_ngr = set(_ngrams(ref_tokens, n).keys())

        p_add = p_ngr - s_ngr
        r_add = r_ngr - s_ngr
        add_correct = p_add & r_add
        p_add_prec = _safe_div(len(add_correct), len(p_add))
        p_add_rec = _safe_div(len(add_correct), len(r_add))
        add_scores.append(_f1(p_add_prec, p_add_rec))

        p_keep = p_ngr & s_ngr
        r_keep = r_ngr & s_ngr
        keep_correct = p_keep & r_keep
        p_keep_prec = _safe_div(len(keep_correct), len(p_keep))
        p_keep_rec = _safe_div(len(keep_correct), len(r_keep))
        keep_scores.append(_f1(p_keep_prec, p_keep_rec))

        p_del = s_ngr - p_ngr
        r_del = s_ngr - r_ngr
        del_correct = p_del & r_del
        del_prec = _safe_div(len(del_correct), len(p_del))
        del_rec = _safe_div(len(del_correct), len(r_del))
        del_scores.append(_f1(del_prec, del_rec))

    sari = (np.mean(add_scores) + np.mean(keep_scores) + np.mean(del_scores)) / 3.0
    return float(sari * 100.0)


def compute_sari(source: str, prediction: str, reference: str) -> float | None:
    source = normalize_spaces(source)
    prediction = normalize_spaces(prediction)
    reference = normalize_spaces(reference)

    if not source or not prediction or not reference:
        return None

    if HAS_EASSE:
        try:
            score = corpus_sari(
                orig_sents=[source],
                sys_sents=[prediction],
                refs_sents=[[reference]],
            )
            return float(score)
        except Exception:
            pass

    return _sari_fallback_single(source, prediction, reference)


# def compute_bertscore_batch(
#     references: list[str],
#     predictions: list[str],
#     lang: str = "es",
#     model_type: str | None = None,
# ) -> list[float | None]:
#     if not HAS_BERTSCORE:
#         return [None] * len(predictions)

#     refs = [normalize_spaces(x) for x in references]
#     preds = [normalize_spaces(x) for x in predictions]

#     valid_mask = [bool(r) and bool(p) for r, p in zip(refs, preds)]
#     if not any(valid_mask):
#         return [None] * len(predictions)

#     valid_preds = [p for p, ok in zip(preds, valid_mask) if ok]
#     valid_refs = [r for r, ok in zip(refs, valid_mask) if ok]

#     try:
#         _, _, f1 = bertscore_score(
#             valid_preds,
#             valid_refs,
#             lang=lang,
#             model_type=model_type,
#             verbose=False,
#             rescale_with_baseline=False,
#         )

#         vals = [None] * len(predictions)
#         j = 0
#         for i, ok in enumerate(valid_mask):
#             if ok:
#                 vals[i] = float(f1[j].item())
#                 j += 1
#         return vals
#     except Exception as e:
#         print("ERROR EN BERTSCORE:", repr(e))
#         return [None] * len(predictions)

def compute_bertscore_batch(
    references: list[str],
    predictions: list[str],
    lang: str = "es",
    model_type: str | None = None,
    batch_size: int = 16,
) -> list[float | None]:
    if not HAS_BERTSCORE:
        return [None] * len(predictions)

    refs = [normalize_spaces(x) for x in references]
    preds = [normalize_spaces(x) for x in predictions]

    valid_mask = [bool(r) and bool(p) for r, p in zip(refs, preds)]
    if not any(valid_mask):
        return [None] * len(predictions)

    vals = [None] * len(predictions)

    valid_indices = [i for i, ok in enumerate(valid_mask) if ok]
    valid_refs = [refs[i] for i in valid_indices]
    valid_preds = [preds[i] for i in valid_indices]

    try:
        for start in range(0, len(valid_preds), batch_size):
            end = start + batch_size

            batch_preds = valid_preds[start:end]
            batch_refs = valid_refs[start:end]
            batch_indices = valid_indices[start:end]

            _, _, f1 = bertscore_score(
                batch_preds,
                batch_refs,
                lang=lang,
                model_type=model_type,
                verbose=False,
                rescale_with_baseline=False,
            )

            for idx_local, idx_global in enumerate(batch_indices):
                vals[idx_global] = float(f1[idx_local].item())

        return vals

    except Exception as e:
        print("ERROR EN compute_bertscore_batch:", repr(e))
        return [None] * len(predictions)


def compute_sbert_similarity_batch(
    sources: list[str],
    predictions: list[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> list[float | None]:
    if not HAS_SBERT:
        return [None] * len(predictions)

    srcs = [normalize_spaces(x) for x in sources]
    preds = [normalize_spaces(x) for x in predictions]

    valid_mask = [bool(s) and bool(p) for s, p in zip(srcs, preds)]
    if not any(valid_mask):
        return [None] * len(predictions)

    valid_srcs = [s for s, ok in zip(srcs, valid_mask) if ok]
    valid_preds = [p for p, ok in zip(preds, valid_mask) if ok]

    try:
        model = SentenceTransformer(model_name)
        emb_src = model.encode(valid_srcs, convert_to_tensor=True, show_progress_bar=False)
        emb_pred = model.encode(valid_preds, convert_to_tensor=True, show_progress_bar=False)

        sims = util.cos_sim(emb_src, emb_pred)
        diag = sims.diag().detach().cpu().numpy().tolist()

        vals = [None] * len(predictions)
        j = 0
        for i, ok in enumerate(valid_mask):
            if ok:
                vals[i] = float(diag[j])
                j += 1
        return vals
    except Exception as e:
        print("ERROR EN SBERT:", repr(e))
        return [None] * len(predictions)


def evaluate_row(
    source: str,
    prediction: str,
    reference: str | None = None,
) -> dict[str, Any]:
    source = normalize_spaces(source)
    prediction = normalize_spaces(prediction)
    reference = normalize_spaces(reference)

    fh_pred = fernandez_huerta(prediction)
    fh_src = fernandez_huerta(source)
    inf_pred = inflesz(prediction)
    inf_src = inflesz(source)

    row = {
        "src_words": word_count(source),
        "pred_words": word_count(prediction),
        "ref_words": word_count(reference),
        "src_sentences": len(split_sentences(source)),
        "pred_sentences": len(split_sentences(prediction)),
        "ref_sentences": len(split_sentences(reference)),
        "sari": compute_sari(source, prediction, reference) if reference else None,
        "bleu": compute_bleu(reference, prediction) if reference else None,
        "fernandez_huerta_pred": fh_pred,
        "fernandez_huerta_src": fh_src,
        "fernandez_huerta_delta": (fh_pred - fh_src) if fh_pred is not None and fh_src is not None else None,
        "compression_ratio_eval": compression_ratio(source, prediction),
        "sentence_splits": sentence_splits(source, prediction),
        "levenshtein_similarity": levenshtein_similarity(source, prediction),
        "exact_copy": exact_copy(source, prediction),
        "additions_proportion": additions_proportion(source, prediction),
        "deletions_proportion": deletions_proportion(source, prediction),
        "inflesz_pred": inf_pred,
        "inflesz_src": inf_src,
        "inflesz_delta": (inf_pred - inf_src) if inf_pred is not None and inf_src is not None else None,
    }

    rouge_scores = compute_rouge(reference, prediction) if reference else {
        "rouge1_f": None,
        "rouge2_f": None,
        "rougeL_f": None,
    }
    row.update(rouge_scores)

    row["bertscore_f1"] = None
    row["sbert_similarity"] = None

    return row


def evaluate_dataframe(
    df: pd.DataFrame,
    source_col: str = "source_text",
    pred_col: str = "generated_text",
    ref_col: str = "reference_text",
    compute_bertscore: bool = True,
    compute_sbert: bool = False,
    bertscore_lang: str = "es",
    bertscore_model_type: str | None = None,
    sbert_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> pd.DataFrame:
    work_df = df.copy()

    row_metrics = []
    for _, r in work_df.iterrows():
        metrics = evaluate_row(
            source=r.get(source_col, ""),
            prediction=r.get(pred_col, ""),
            reference=r.get(ref_col, ""),
        )
        row_metrics.append(metrics)

    metrics_df = pd.DataFrame(row_metrics)
    out_df = pd.concat([work_df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)

    if compute_bertscore:
        bert_vals = compute_bertscore_batch(
            references=out_df[ref_col].fillna("").astype(str).tolist(),
            predictions=out_df[pred_col].fillna("").astype(str).tolist(),
            lang=bertscore_lang,
            model_type=bertscore_model_type,
        )
        out_df["bertscore_f1"] = bert_vals

    if compute_sbert:
        sbert_vals = compute_sbert_similarity_batch(
            sources=out_df[source_col].fillna("").astype(str).tolist(),
            predictions=out_df[pred_col].fillna("").astype(str).tolist(),
            model_name=sbert_model_name,
        )
        out_df["sbert_similarity"] = sbert_vals

    return out_df


def summarize_metrics(
    df_eval: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    metric_cols = [
        "sari",
        "bleu",
        "fernandez_huerta_pred",
        "fernandez_huerta_src",
        "fernandez_huerta_delta",
        "compression_ratio_eval",
        "sentence_splits",
        "levenshtein_similarity",
        "exact_copy",
        "additions_proportion",
        "deletions_proportion",
        "rouge1_f",
        "rouge2_f",
        "rougeL_f",
        "inflesz_pred",
        "inflesz_src",
        "inflesz_delta",
        "bertscore_f1",
        "sbert_similarity",
    ]

    metric_cols = [c for c in metric_cols if c in df_eval.columns]

    if not group_cols:
        return df_eval[metric_cols].mean(numeric_only=True).to_frame().T

    return df_eval.groupby(group_cols)[metric_cols].mean(numeric_only=True).reset_index()