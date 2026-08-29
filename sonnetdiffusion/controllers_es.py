"""Controladores analíticos de formato y rima para el muestreo de sonetos.

Las penalizaciones se calculan sobre pred_xstart convertido a soft tokens
via lm_head: probs = softmax(model.get_logits(x0)). Los gradientes fluyen
hacia la muestra de difusión; langevin_fn aplica K pasos Adagrad que
minimizan λ_fmt·formato + λ_rhyme·rima + λ_rep·repetición + λ_syll·sílaba.
"""
import os
import sys
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rhyme_es import build_suffix_matrix, RHYME_GROUPS, count_syllables_word

EPS = 1e-8

STOPWORD_IDS = None

STOPWORDS = frozenset([
    "de", "del", "el", "la", "los", "las", "un", "una", "y", "e", "o", "u",
    "en", "a", "al", "por", "con", "sin", "que", "se", "su", "sus", "lo",
    "le", "les", "me", "te", "nos", "os", "mi", "tu", "no", "ni", "si",
    "ya", "más", "mas", "tan", "muy", "como", "pero", "sino", "pues",
    "es", "ha", "he",
])


def _build_stopword_ids(vocab):
    global STOPWORD_IDS
    if STOPWORD_IDS is None:
        STOPWORD_IDS = frozenset(vocab[w] for w in STOPWORDS if w in vocab)
    return STOPWORD_IDS


def soft_tokens(model, x0, temperature=1.0):
    """x0 [B, L, d] → probs [B, L, V] via lm_head compartido."""
    logits = model.get_logits(x0)
    return F.softmax(logits / temperature, dim=-1), logits


# =============================================================================
# ALGORITMO 6a: FormatPenalty
# Penaliza desviación del conteo de 14 <eos> (separadores de verso).
# Normalizado por target² para mantener ~O(1).
# =============================================================================
def format_penalty(probs, eos_id, target=14):
    soft_count = probs[:, :, eos_id].sum(dim=1)
    return (((soft_count - float(target)) / float(target)) ** 2).mean()


_SYLL_VEC = None


def _build_syll_vec(vocab, V_model, device):
    """Precomputa conteo de silabas por palabra del vocabulario."""
    global _SYLL_VEC
    if _SYLL_VEC is not None and _SYLL_VEC.device == device:
        return _SYLL_VEC
    sv = torch.zeros(V_model, device=device)
    for word, wid in vocab.items():
        if wid >= V_model:
            continue
        if word in {"eos", "UNK", "PAD", "START", "END", ",", ".", ":", ";"}:
            continue
        sv[wid] = float(count_syllables_word(word))
    _SYLL_VEC = sv
    return _SYLL_VEC


def syllable_penalty(probs, logits, eos_id, syll_vec, target=11):
    """Penaliza versos cuyo conteo soft de silabas se aleja del endecasilabo."""
    B, L, V = probs.shape
    sv = syll_vec[:V]
    hard = logits.argmax(dim=-1).detach()
    total = probs.new_zeros(())
    n = 0
    for b in range(B):
        eos_pos = (hard[b] == eos_id).nonzero(as_tuple=True)[0].tolist()
        prev = 0
        for pe in eos_pos[:14]:
            if pe <= prev:
                prev = pe + 1
                continue
            verse_probs = probs[b, prev:pe]
            soft_count = (verse_probs @ sv).sum()
            total = total + ((soft_count - target) / target) ** 2
            n += 1
            prev = pe + 1
    return total / max(1, n)


def _verse_positions(hard_b, is_rhyme, eos_id, window, max_lines=14,
                     stopword_ids=None):
    """Posicion de la ultima palabra con rima antes de cada <eos>.
    Salta palabras funcion (de, el, y, que...) para apuntar a palabras de contenido."""
    eos_pos = (hard_b == eos_id).nonzero(as_tuple=True)[0].tolist()
    stop = stopword_ids or frozenset()
    pos_map = {}
    for li, pe in enumerate(eos_pos[:max_lines], start=1):
        for p in range(pe - 1, max(-1, pe - window) - 1, -1):
            if p < 0:
                break
            wid = hard_b[p].item()
            if wid in stop:
                continue
            if bool(is_rhyme[wid]):
                pos_map[li] = p
                break
    return pos_map


# =============================================================================
# ALGORITMO 6b: RhymePenalty
# Penaliza discordancia de sufijo fonético dentro de cada grupo ABBA ABBA CDE CDE.
# Dos palabras distintas con igual sufijo dan el mismo q → premia rima, NO repetición.
# =============================================================================
def rhyme_penalty(probs, logits, S, eos_id, groups=RHYME_GROUPS, window=6,
                  stopword_ids=None):
    B = probs.shape[0]
    hard = logits.argmax(dim=-1).detach()
    is_rhyme = (S.sum(dim=1) > 0)
    total = probs.new_zeros(())
    n = 0
    for b in range(B):
        pos_map = _verse_positions(hard[b], is_rhyme, eos_id, window,
                                   stopword_ids=stopword_ids)
        q = {}
        for li, p in pos_map.items():
            qd = probs[b, p] @ S
            s = qd.sum()
            if s > EPS:
                q[li] = qd / (s + EPS)
        for lines in groups.values():
            present = [q[l] for l in lines if l in q]
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    total = total + (1.0 - (present[i] * present[j]).sum())
                    n += 1
    return total / max(1, n)


# =============================================================================
# ALGORITMO 6c: RepetitionPenalty
# Penaliza la MISMA palabra en posición final de verso dentro de cada grupo.
# Usa overlap al cuadrado ⟨pᵢ,pⱼ⟩² para gradiente más fuerte cerca de convergencia.
# =============================================================================
def repetition_penalty(probs, logits, S, eos_id, groups=RHYME_GROUPS, window=6,
                       stopword_ids=None):
    B = probs.shape[0]
    hard = logits.argmax(dim=-1).detach()
    is_rhyme = (S.sum(dim=1) > 0)
    total = probs.new_zeros(())
    n = 0
    for b in range(B):
        pos_map = _verse_positions(hard[b], is_rhyme, eos_id, window,
                                   stopword_ids=stopword_ids)
        for lines in groups.values():
            present = [probs[b, pos_map[l]] for l in lines if l in pos_map]
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    overlap = (present[i] * present[j]).sum()
                    total = total + overlap ** 2
                    n += 1
    return total / max(1, n)


# =============================================================================
# ALGORITMO 6 (núcleo): Corrección Langevin
# K pasos Adagrad sobre pred_xstart en cada paso de denoising DDIM.
# loss = λ_fmt·fmt + λ_rhyme·rhy + λ_rep·rep + λ_syll·syl + regularizacion
# Se conecta al sampler via ddim_sample_loop(..., langevin_fn=...).
# =============================================================================
def concept_presence_penalty(probs, concept_ids, eos_id, weights=None):
    """Penaliza ausencia de palabras concepto en el soneto.
    Usa -log(max_prob) en vez de 1-max_prob para gradiente mas fuerte."""
    B, L, V = probs.shape
    eos_mask = torch.zeros(L, device=probs.device)
    hard = probs.argmax(dim=-1).detach()
    for b in range(B):
        eos_mask = eos_mask + (hard[b] == eos_id).float()
    non_eos = (eos_mask < B * 0.5).float().unsqueeze(0).unsqueeze(-1)
    masked_probs = probs * non_eos
    total = probs.new_zeros(())
    total_weight = 0.0
    for cid in concept_ids:
        w = weights.get(cid, 1.0) if weights else 1.0
        max_prob = masked_probs[:, :, cid].max(dim=1).values.mean()
        total = total + w * (-torch.log(max_prob + 1e-8)).clamp(max=5.0)
        total_weight += w
    return total / max(1e-8, total_weight)


_VOCAB_SIGLIP_CACHE = None


def _get_vocab_siglip(vocab, encoder):
    """Codifica todo el vocabulario por SigLIP (cacheado tras primera llamada)."""
    global _VOCAB_SIGLIP_CACHE
    if _VOCAB_SIGLIP_CACHE is not None:
        return _VOCAB_SIGLIP_CACHE

    id2word = {v: k for k, v in vocab.items()}
    skip = {"eos", "UNK", "PAD", "START", "END", ",", ".", ":", ";"}
    skip.update(STOPWORDS)
    skip_ids = {vocab[s] for s in skip if s in vocab}

    candidates = []
    candidate_ids = []
    for wid in sorted(id2word.keys()):
        word = id2word[wid]
        if wid in skip_ids or not word.isalpha() or len(word) < 3:
            continue
        candidates.append(word)
        candidate_ids.append(wid)

    print(f"[SigLIP] Codificando {len(candidates)} palabras del vocabulario...")
    batch_size = 64
    vecs = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        batch_vecs = torch.stack([encoder.encode_text(w) for w in batch])
        vecs.append(batch_vecs)
    cand_vecs = torch.cat(vecs, dim=0)
    cand_norm = F.normalize(cand_vecs, dim=1)
    print(f"[SigLIP] Codificacion completa.")

    _VOCAB_SIGLIP_CACHE = (candidates, candidate_ids, cand_norm)
    return _VOCAB_SIGLIP_CACHE


def expand_concepts_by_siglip(concept_words, vocab, encoder, top_k=3,
                              sim_threshold=0.3):
    """Expande conceptos usando embeddings de texto SigLIP 768-d.
    Conceptos directos peso=2.0, vecinos peso=1.0."""
    candidates, candidate_ids, cand_norm = _get_vocab_siglip(vocab, encoder)

    expanded_ids = set()
    concept_neighborhoods = {}
    weights = {}

    for word in concept_words:
        if word not in vocab:
            continue
        wid = vocab[word]
        expanded_ids.add(wid)
        weights[wid] = 2.0

        w_vec = F.normalize(encoder.encode_text(word), dim=0)
        sims = cand_norm @ w_vec
        top_vals, top_idx = sims.topk(top_k + 1)
        neighbors = []
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            cid = candidate_ids[idx]
            cw = candidates[idx]
            if cw == word or val < sim_threshold:
                continue
            if len(neighbors) >= top_k:
                break
            expanded_ids.add(cid)
            if cid not in weights:
                weights[cid] = 1.0
            neighbors.append(cw)
        concept_neighborhoods[word] = neighbors

    return list(expanded_ids), concept_neighborhoods, weights


def make_langevin_fn(diffusion, model, S, eos_id, *, lambda_fmt=0.05,
                     lambda_rhyme=0.5, lambda_rep=0.2, lambda_syll=0.05,
                     coef=0.0005,
                     step_size=0.03, K=5, t_stop=10, temperature=0.8, window=6,
                     groups=RHYME_GROUPS, debug=False, vocab=None,
                     concept_word_ids=None, lambda_concept=0.15,
                     concept_weights=None,
                     emotion_word_ids=None, lambda_emotion=0.15):
    dev = next(model.parameters()).device
    S = S.to(dev)
    stop_ids = _build_stopword_ids(vocab) if vocab else frozenset()
    _syll_built = [False]

    def langevin_fn(sample, mean, sigma, alpha, t, prev_sample):
        if t[0].item() < t_stop:
            return sample
        input_embs = torch.nn.Parameter(sample)
        with torch.enable_grad():
            for i in range(K):
                opt = torch.optim.Adagrad([input_embs], lr=step_size)
                opt.zero_grad()
                out = diffusion.p_mean_variance(
                    model, input_embs, t, clip_denoised=False,
                    denoised_fn=None, model_kwargs={})
                x0 = out["pred_xstart"]
                probs, logits = soft_tokens(model, x0, temperature)
                fmt = format_penalty(probs, eos_id)
                rhy = rhyme_penalty(probs, logits, S, eos_id, groups, window,
                                    stopword_ids=stop_ids)
                rep = repetition_penalty(probs, logits, S, eos_id, groups, window,
                                         stopword_ids=stop_ids)
                syl = probs.new_zeros(())
                if vocab is not None:
                    if not _syll_built[0]:
                        _syll_built[0] = True
                        _build_syll_vec(vocab, probs.shape[-1], dev)
                    syl = syllable_penalty(probs, logits, eos_id, _SYLL_VEC)
                sigma_safe = sigma.clamp(min=0.1)
                logp = coef * ((mean - input_embs) ** 2 / sigma_safe).mean(dim=0).sum()
                loss = (lambda_fmt * fmt + lambda_rhyme * rhy
                        + lambda_rep * rep + lambda_syll * syl + logp)
                cpt_val = 0.0
                if concept_word_ids is not None:
                    cpt = concept_presence_penalty(probs, concept_word_ids, eos_id,
                                                   weights=concept_weights)
                    loss = loss + lambda_concept * cpt
                    cpt_val = cpt.item()
                emo_val = 0.0
                if emotion_word_ids is not None:
                    emo = concept_presence_penalty(probs, emotion_word_ids, eos_id)
                    loss = loss + lambda_emotion * emo
                    emo_val = emo.item()
                loss.backward()
                opt.step()
                if debug and i == K - 1:
                    print(f"  [ctrl t={t[0].item()}] fmt={fmt.item():.3f} "
                          f"rhy={rhy.item():.3f} rep={rep.item():.3f} "
                          f"syl={syl.item():.3f} "
                          f"cpt={cpt_val:.3f} emo={emo_val:.3f}")
                input_embs = torch.nn.Parameter(input_embs.data.detach())
        return input_embs.data

    return langevin_fn

