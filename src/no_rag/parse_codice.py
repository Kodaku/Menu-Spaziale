import json
import os
import sys
import time
from pathlib import Path
import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import API_KEY_ENV_NAME, GROQ_MODEL_LLAMA_VERSATILE

from groq import Groq

CODICE_PATH = Path("../../Dataset/knowledge_base_md/codice_galattico/Codice Galattico.md")
KB_DIR = Path("../kb")


# Quelli che seguono sono prompt suggeriti da Claude che ha avuto conoscenza dell'intera kb. Per questo motivo sono presenti valori molto precisi. Si considera che a disposizione c'è un LLM non potentissimo (Llama3.3-70b) utilzizato tramite API di Groq. Non potentissimo in senso di potenzialità limitate per via dei limiti di utilizzo dell'API. In ogni caso per limitare le allucinazioni Claude ha costruito un prompt quanto più preciso possibile.
# TODO: questo prompt tuttavia non è molto flessibile. Cambio di requisiti (ad esempio le domande o altri/nuovi documenti legati al codice galattico) o di struttura dello stesso Codice Galattico => cambio del prompt (e forse anche di altro).
SYSTEM_PROMPT = (
    "You are an expert data extractor. Return only valid JSON with no extra text."
)

LIMITS_PROMPT = """Extract ALL regulated substances and their quantity limits from this text.

The text is from "Codice Galattico" (galactic food safety code) and has OCR artifacts
(e.g., parentheses replacing 'ff', 'MuNa' instead of 'Muffa', etc.).

The substances listed in the table (section 2.2) and their limits (section 3) are:

Rules from section 3:
- CRP > 0.90 → limit 0.5% of total mass
- CRP 0.65–0.90 → limit 1%
- IPM > 0.90 → limit 4%
- IPM ≤ 0.90 → limit 5% (general max for all regulated substances)
- IBX > 0.70 AND μ > 0.50 → limit 0.1%
- IBX > 0.70 AND μ ≤ 0.50 → limit 0.25%
- δQ > 0.30 → limit 3%
- δQ ≤ 0.30 → limit 5% (general max)
- CDT > 0.70 → limit 2%
- CDT ≤ 0.70 → limit 3%
- All regulated substances: general max 5%
- For each substance, use the MOST RESTRICTIVE applicable limit

IMPORTANT name corrections (OCR artifacts):
- "MuNa Lunare" → "Muffa Lunare"
- "A(umicatura" → "Affumicatura"

Return JSON with this structure:
{
  "substances": [
    {
      "name": "exact corrected ingredient name",
      "category": "Psicotrope|Mitica|Xenobiologica|Quantica|Spazio-Temporale",
      "properties": {"CRP": 0.89, "IEI": 0.3},
      "limit_pct": 1.0
    }
  ]
}

TEXT:
""" + CODICE_PATH.read_text(encoding="utf-8")[:4000]

LICENZE_PROMPT = """Extract ALL cooking technique → required license mappings from the section 4 of this Codice Galattico text.

The text has OCR artifacts. Section 4 covers:
4.1 Marinatura, 4.2 Affumicatura, 4.3 Fermentazione, 4.4 Decostruzione,
4.5 Sferificazione (section title may be blank/corrupted), 4.6 Taglio,
4.7 Impasto, 4.8 Surgelamento, 4.9 Cottura (with sub-sections for Bollitura,
Grigliatura, Cottura al Forno, Cottura al Vapore, Cottura Sottovuoto, Cottura al Salto)

Some entries have OCR-dropped technique names (line starts with "- livello..." or "- (Q)..." or similar).
For those, infer the technique name from the ORDER within the section using this ordering:

Section 4.1 Marinatura techniques (in order):
1. Marinatura a Infusione Gravitazionale
2. Marinatura Temporale Sincronizzata
3. Marinatura Psionica
4. Marinatura tramite Reazioni d'Antimateria Diluite
5. Marinatura Sotto Zero a Polarità Inversa

Section 4.2 Affumicatura techniques (in order):
1. Affumicatura a Stratificazione Quantica
2. Affumicatura Temporale Risonante
3. Affumicatura Psionica Sensoriale
4. Affumicatura tramite Big Bang Microcosmico
5. Affumicatura Polarizzata a Freddo Iperbarico

Section 4.3 Fermentazione techniques (in order):
1. Fermentazione Quantica a Strati Multiversali
2. Fermentazione Temporale Sincronizzata
3. Fermentazione Psionica Energetica
4. Fermentazione tramite Singolarità
5. Fermentazione Quantico Biometrica

Section 4.4 Decostruzione techniques (in order):
1. Decostruzione Atomica a Strati Energetici
2. Decostruzione Magnetica Risonante
3. Decostruzione Bio-Fotonica Emotiva
4. Decostruzione Ancestrale
5. Decostruzione Interdimensionale Lovecraftiana

Section 4.5 Sferificazione techniques (in order, section title may be missing):
1. Sferificazione a Gravità Psionica Variabile
2. Sferificazione Filamentare a Molecole Vibrazionali
3. Sferificazione Cromatica Interdimensionale
4. Sferificazione con Campi Magnetici Entropici
5. Sferificazione tramite Matrici Biofotiche

Section 4.6 Taglio techniques (in order):
1. Taglio Dimensionale a Lame Fotofiliche
2. Taglio a Risonanza Sonica Rigenerativa
3. Affettamento a Pulsazioni Quantistiche
4. Taglio Sinaptico Biomimetico
5. Incisione Elettromagnetica Plasmica

Section 4.7 Impasto techniques (in order):
1. Impasto Gravitazionale Vorticoso
2. Amalgamazione Sintetica Molecolare
3. Impasto a Campi Magnetici Dualistici
4. Sinergia Elettro-Osmotica Programmabile
5. Modellatura Onirica Tetrazionale

Section 4.8 Surgelamento techniques (in order):
1. Cryo-Tessitura Energetica Polarizzata
2. Congelamento Bio-Luminiscente Sincronico
3. Cristallizzazione Temporale Reversiva
4. Congelazione Iperdimensionalmente Stratificata
5. Surgelamento Antimaterico a Risonanza Inversa

Section 4.9.1 Bollitura (in order):
1. Ebollizione Magneto-Cinetica Pulsante
2. Bollitura Infrasonica Armonizzata
3. Bollitura Termografica a Rotazione Veloce
4. Bollitura Entropica Sincronizzata
5. Idro-Cristallizzazione Sonora Quantistica

Section 4.9.2 Grigliatura (in order):
1. Grigliatura a Energia Stellare DiV
2. Grigliatura Plasma Sintetico Risonante
3. Grigliatura Eletro-Molecolare a Spaziatura Variabile
4. Grigliatura Tachionica Refrattaria
5. Grigliatura Psionica Dinamica Ritmica

Section 4.9.3 Cottura al Forno (in order):
1. Cottura al Forno con Paradosso Temporale Cronospeculare
2. Cottura con Microonde Entropiche Sincronizzate
3. Cottura a Forno Dinamico Inversionale
4. Cottura Olografica Quantum Fluttuante
5. Cottura Geomagnetica Psicosincronizzata

Section 4.9.4 Cottura al Vapore (only these 2 have requirements):
1. Cottura a Vapore Risonante Simbiotico
2. Cottura Idrodinamica Autoregolante

Section 4.9.5 Cottura Sottovuoto (in order):
1. Cottura Sottovuoto Antimateria
2. Cottura Sottovuoto Multirealità Collassante
3. Cottura Sottovuoto Frugale Energeticamente Negativa
4. Cottura Sottovuoto Pulsar Magnetica
5. Cottura Sottovuoto Bioma Sintetico

Section 4.9.6 Cottura al Salto (in order, "Saltare in Padella Classica" has NO requirement):
1. Saltare in Padella Big Bang Termico
2. Saltare in Padella Realtà Energetiche Parallele
3. Saltare in Padella Singolarità Inversa
4. Saltare in Padella Sinergia Psionica

License type keys to use: "P" (Psionica), "t" (Temporale), "G" (Gravitazionale),
"e+" (Antimateria), "Mx" (Magnetica), "Q" (Quantistica), "c" (Luce), "LTK"
All levels are integers (convert Roman numerals: I=1, II=2, III=3, IV=4, V=5, VI=6).
Special: "VI+" → 7.

If a technique has NO listed requirements, do NOT include it (it only needs default licenses).

Return JSON:
{
  "techniques": [
    {
      "name": "exact technique name",
      "licenses": {"G": 2},
      "ltk": 2
    }
  ]
}

TEXT:
"""


def call_groq(client: Groq, prompt: str, model: str = "llama-3.3-70b-versatile") -> dict:
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt < 2:
                print(f"  Retry {attempt + 1}/3 after error: {e}")
                time.sleep(5)
            else:
                raise


def build_ingrediente_limiti(client: Groq) -> None:
    print("Estrazione limiti ingredienti dal Codice Galattico...")
    prompt = LIMITS_PROMPT

    result = call_groq(client, prompt, model=GROQ_MODEL_LLAMA_VERSATILE)
    # Per limiti di utilizzo delle sostanze
    substances = result.get("substances", [])

    # Costruzione di una kb per limiti di utilizzo delle sostanze
    limiti: dict[str, dict] = {}
    for s in substances:
        name = s.get("name", "").strip()
        if name:
            limiti[name] = {
                "category": s.get("category", ""),
                "properties": s.get("properties", {}),
                "limit_pct": s.get("limit_pct", 5.0),
            }

    out_path = KB_DIR / "ingrediente_limiti.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(limiti, f, indent=2, ensure_ascii=False)
    print(f"Salvati {len(limiti)} limiti ingredienti in {out_path}")


def build_tecnica_to_licenze(client: Groq) -> None:
    print("Estrazione dei requisiti di tecniche di licenza dal Codice Galattico...")
    codice_text = CODICE_PATH.read_text(encoding="utf-8")

    # Dove si trova esattamente questa sezione del codice galattico
    # è stato merito di Claude.
    # TODO: Qui c'è una ricerca per test hardcoded case sensitive => cambio del codice galattico => perdita della capacità di parsing. Per ora funziona per un MVP
    lines = codice_text.splitlines()
    section4_start = 0
    for i, line in enumerate(lines):
        if "4 Licenze e Tecniche" in line or "- 4 " in line:
            section4_start = i
            break

    section4_text = "\n".join(lines[section4_start:])

    prompt = LICENZE_PROMPT + section4_text

    result = call_groq(client, prompt)
    techniques = result.get("techniques", [])

    # Costruzione dizionario di tecniche e requisiti di licenze
    licenze: dict[str, dict] = {}
    for t in techniques:
        name = t.get("name", "").strip()
        if name:
            entry: dict = {}
            lics = t.get("licenses", {})
            if lics:
                entry["licenses"] = lics
            ltk = t.get("ltk")
            if ltk:
                entry["ltk"] = ltk
            if entry:
                licenze[name] = entry

    out_path = KB_DIR / "tecnica_to_licenze.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(licenze, f, indent=2, ensure_ascii=False)
    print(f"Salvate {len(licenze)} technique→license mappings in {out_path}")


def build_codice_artifacts() -> None:
    api_key = os.environ.get(API_KEY_ENV_NAME)
    if not api_key:
        raise ValueError("Variabile d'ambiente GROQ_API_KEY non settata")

    client = Groq(api_key=api_key)
    KB_DIR.mkdir(exist_ok=True)

    build_ingrediente_limiti(client)
    time.sleep(3)
    build_tecnica_to_licenze(client)


if __name__ == "__main__":
    dotenv.load_dotenv()
    build_codice_artifacts()
