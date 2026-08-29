"""Vocabulario de conceptos visuales, emociones y mapping pysentimiento.

VISUAL_CONCEPTS se rankean por similitud coseno SigLIP contra la imagen
para producir los top-K candidatos en HCI-1. Cubren objetos, escenas,
luz, color y temas abstractos centrales en la tradición sonetista española.

EMOTION_LABELS se rankean igual para HCI-2.
"""

# ── Naturaleza ──────────────────────────────────────────────────────
_NATURALEZA = [
    "árbol", "bosque", "montaña", "río", "mar", "océano", "playa", "arena",
    "roca", "piedra", "flor", "rosa", "lirio", "clavel", "jardín", "hoja",
    "rama", "raíz", "hierba", "prado", "campo", "valle", "colina", "desierto",
    "selva", "cueva", "lago", "cascada", "manantial", "isla",
    "cielo", "nube", "sol", "luna", "estrella", "lluvia", "nieve", "niebla",
    "fuego", "llama", "humo", "viento", "tormenta", "relámpago", "arcoíris",
    "agua", "tierra", "hielo", "ceniza", "polvo",
]

# ── Fauna ───────────────────────────────────────────────────────────
_FAUNA = [
    "pájaro", "ave", "águila", "paloma", "cisne", "ruiseñor", "golondrina",
    "caballo", "perro", "gato", "lobo", "león", "cordero",
    "mariposa", "pez", "serpiente", "ciervo", "toro", "oveja", "abeja",
]

# ── Lugares / arquitectura ──────────────────────────────────────────
_LUGARES = [
    "ciudad", "pueblo", "calle", "plaza", "iglesia", "catedral", "templo",
    "torre", "puente", "castillo", "palacio", "ruina", "muro", "ventana",
    "puerta", "camino", "sendero", "puerto", "barco", "vela", "faro",
    "molino", "cabaña", "tumba", "sepulcro", "altar", "trono", "cárcel",
]

# ── Tiempo / luz ────────────────────────────────────────────────────
_TIEMPO_LUZ = [
    "amanecer", "atardecer", "anochecer", "mediodía", "crepúsculo", "alba",
    "aurora", "ocaso", "penumbra", "sombra", "luz", "claridad", "oscuridad",
    "noche", "día", "invierno", "otoño", "primavera", "verano",
    "hora", "instante", "eternidad", "tiempo",
]

# ── Personas / cuerpo ──────────────────────────────────────────────
_PERSONAS = [
    "mujer", "hombre", "niño", "niña", "anciano", "rostro", "ojos", "manos",
    "cabello", "labios", "pecho", "corazón", "sangre", "lágrima",
    "sonrisa", "silueta", "multitud", "amante", "madre", "guerrero",
]

# ── Objetos / símbolos ─────────────────────────────────────────────
_OBJETOS = [
    "espada", "corona", "cruz", "cadena", "llave", "anillo", "espejo",
    "cristal", "metal", "madera", "tela", "libro", "pluma", "vela",
    "copa", "campana", "bandera", "reloj", "máscara", "escudo",
]

# ── Colores / cualidades sensoriales ───────────────────────────────
_CUALIDADES = [
    "luminoso", "oscuro", "brillante", "tenue", "vasto", "diminuto",
    "antiguo", "salvaje", "sereno", "turbulento", "frío", "cálido",
    "húmedo", "seco", "denso", "etéreo", "rugoso", "suave",
    "dorado", "plateado", "azul", "rojo", "verde", "gris",
    "blanco", "negro", "púrpura", "ocre", "carmesí",
]

# ── Temas abstractos (centrales en el soneto español) ──────────────
_ABSTRACTOS = [
    # amor / deseo
    "amor", "pasión", "deseo", "ternura", "celos", "seducción",
    "intimidad", "abrazo", "beso", "entrega",
    # muerte / fugacidad
    "muerte", "olvido", "ceniza", "ruina", "decadencia", "fin",
    "fugacidad", "vanidad",
    # existencia / alma
    "vida", "alma", "destino", "fortuna", "gloria", "fe",
    "virtud", "pecado", "culpa", "redención", "sacrificio",
    # dolor / conflicto
    "dolor", "pena", "llanto", "sufrimiento", "guerra", "herida",
    "soledad", "abandono", "exilio", "cautiverio",
    # contemplación / trascendencia
    "silencio", "calma", "quietud", "paz", "armonía",
    "belleza", "misterio", "asombro", "sueño", "visión",
    # movimiento / cambio
    "viaje", "partida", "regreso", "encuentro", "despedida",
    "renacer", "despertar", "caída", "vuelo", "naufragio",
    # sentimiento-escena
    "melancolía", "nostalgia", "alegría", "tristeza", "esperanza",
    "miedo", "fuerza", "fragilidad", "grandeza", "vacío",
    "tensión", "refugio", "memoria", "suspiro",
    # espacio
    "horizonte", "profundidad", "altura", "distancia", "umbral",
    "frontera", "abismo", "cumbre", "reflejo",
]

VISUAL_CONCEPTS = list(dict.fromkeys(
    _NATURALEZA + _FAUNA + _LUGARES + _TIEMPO_LUZ +
    _PERSONAS + _OBJETOS + _CUALIDADES + _ABSTRACTOS
))

# ── Emociones (Ekman directo) ──────────────────────────────────────
# 6 categorias Ekman + neutral, alineadas con pysentimiento/robertuito-emotion-analysis.

EMOTION_LABELS_EKMAN = {
    "alegria": "joy",
    "tristeza": "sadness",
    "ira": "anger",
    "miedo": "fear",
    "sorpresa": "surprise",
    "asco": "disgust",
    "neutral": "others",
}

EMOTION_LABELS = list(EMOTION_LABELS_EKMAN.keys())


EMOTION_VOCAB = {
    "alegria": [
        "alegría", "gozo", "dicha", "risa", "júbilo", "contento", "fiesta",
        "deleite", "placer", "dulzura", "encanto", "feliz", "dichoso",
        "gozoso", "radiante", "celebrar", "cantar", "danzar", "reír",
    ],
    "tristeza": [
        "dolor", "pena", "llanto", "lágrima", "muerte", "olvido", "soledad",
        "lamento", "duelo", "angustia", "melancolía", "sufrir", "llorar",
        "morir", "triste", "amargo", "sombrío", "oscuro", "desolado",
    ],
    "ira": [
        "furia", "rabia", "odio", "venganza", "rencor", "ira", "fuego",
        "llama", "destruir", "maldito", "furioso", "cruel", "feroz",
        "arder", "quemar", "espada", "guerra", "sangre", "golpe",
    ],
    "miedo": [
        "miedo", "temor", "espanto", "horror", "sombra", "abismo",
        "oscuridad", "tiniebla", "huir", "temblar", "peligro",
        "amenaza", "terrible", "pálido", "frío", "noche", "fantasma",
    ],
    "sorpresa": [
        "asombro", "prodigio", "misterio", "maravilla", "portento",
        "milagro", "extraño", "insólito", "repentino", "súbito",
        "descubrir", "revelar", "enigma", "secreto", "oculto",
    ],
    "asco": [
        "repugnancia", "desprecio", "miseria", "podredumbre", "inmundo",
        "vergüenza", "infame", "abyecto", "sucio", "corromper",
        "pudrir", "hedor", "gusano", "traición", "vileza",
    ],
    "neutral": [],
}


def pysent_score_for_emotion(probas: dict, target_emotion: str) -> float:
    """Probabilidad directa de la categoria Ekman correspondiente."""
    ekman_key = EMOTION_LABELS_EKMAN.get(target_emotion, target_emotion)
    return probas.get(ekman_key, 0.0)
