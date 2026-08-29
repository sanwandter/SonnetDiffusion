"""Encoder multimodal SigLIP 2 para Vision2Sonnet.

Proyecta imágenes y texto al espacio compartido de 768-d de
SigLIP 2 (google/siglip2-base-patch16-224).
"""
import torch
import torch.nn.functional as F
from vocab_es import VISUAL_CONCEPTS, EMOTION_LABELS


def _rank(v, labels, label_vecs, top_k):
    """Rankea etiquetas por similitud coseno contra un vector de referencia."""
    vn = F.normalize(v, dim=0)
    ln = F.normalize(label_vecs, dim=1)
    scores = ln @ vn
    order = scores.argsort(descending=True)[:top_k]
    return [(labels[i], float(scores[i])) for i in order]


class SiglipEncoder:
    def __init__(self, model_name="google/siglip2-base-patch16-224", device=None):
        from transformers import AutoModel, AutoProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self._concept_vecs = None
        self._emotion_vecs = None

    # =========================================================================
    # ALGORITMO 2: EncodeImage → v_img
    # Entrada: imagen RGB  |  Salida: embedding normalizado (768-d)
    # =========================================================================
    @torch.no_grad()
    def encode_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.get_image_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        return F.normalize(feats[0], dim=0).cpu()

    # =========================================================================
    # ALGORITMO 3 (auxiliar): EncodeEmotion → v_emo
    # Codifica texto al mismo espacio compartido que v_img
    # =========================================================================
    @torch.no_grad()
    def encode_text(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding="max_length",
                                max_length=64, truncation=True).to(self.device)
        out = self.model.get_text_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        return F.normalize(feats[0], dim=0).cpu()

    def _encode_labels(self, labels):
        return torch.stack([self.encode_text(l) for l in labels])

    # =========================================================================
    # ALGORITMO 3: ExtractVisualConcepts → concepts[]
    # Rankea 269 conceptos del vocabulario español contra v_img (sim. coseno)
    # =========================================================================
    def extract_visual_concepts(self, v_img, top_k=15):
        if self._concept_vecs is None:
            self._concept_vecs = self._encode_labels(VISUAL_CONCEPTS)
        return _rank(v_img, VISUAL_CONCEPTS, self._concept_vecs, top_k)

    # =========================================================================
    # ALGORITMO 4: ExtractEmotionCandidates → emotions[]
    # Rankea 7 emociones (Ekman) contra v_img por similitud multimodal
    # =========================================================================
    def extract_emotion_candidates(self, v_img, top_n=5):
        if self._emotion_vecs is None:
            self._emotion_vecs = self._encode_labels(EMOTION_LABELS)
        return _rank(v_img, EMOTION_LABELS, self._emotion_vecs, top_n)
