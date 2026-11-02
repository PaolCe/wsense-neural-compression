"""
hybrid_codec.py — il paradigma del "decoder generativo" (Cap. 2 del deck
"Neural Visual Compression Evolution"), applicato a questa challenge.

IDEA, e perche' batte l'autoencoder end-to-end SU QUESTA METRICA:

    S = w1*PSNR/PSNR_ref + w2*MS-SSIM/MS-SSIM_ref + w3*ln(CR)/ln(CR_ref)

Il termine del rate ha il peso maggiore (0.4 secondo la slide "Kaggle Metric", 1/3 nella
lettura della pagina Overview) ma entra tramite un logaritmo, mentre PSNR e MS-SSIM
entrano linearmente. Dimezzare il payload vale ~6.6 dB di PSNR (~4.9 dB a pesi uguali):
molto piu' di quanto costi. L'ottimo sta quindi a CR molto alto.

Un autoencoder end-to-end NEGOZIA un compromesso rate/qualita' fissato in training. Il
paradigma del Cap. 2 invece: si comprime in modo brutale con un codec classico (AVIF/WebP,
gia' allo stato dell'arte come entropy coder, molto meglio di quello che possiamo allenare
noi in poche ore) e si RECUPERA la qualita' con una rete di restauro sul lato decoder. Si
massimizza il termine del rate e si ricompra la qualita' persa, invece di mediare.

Il deck: "modern learned codecs transmit a tiny, highly compressed visual prompt or a
heavily downsampled latent representation. Models like TinySR and AdcSR are deployed at
the receiver's end (the decoder side)."

Misura sui dati di validazione, senza NESSUNA rete: AVIF a scale 0.35 / q 50 da' gia'
S~1.06 contro lo 0.845 del nostro autoencoder allenato. Il restauro parte da li'.

CONTRATTO: il restauro sta nel modello condiviso (non si paga per immagine); nel payload
viaggiano solo i byte del codec classico piu' un header di 9 byte.
"""
import io
import struct
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# --------------------------------------------------------------------------------------
# metrica
# --------------------------------------------------------------------------------------
SCORE_WEIGHTS = (0.3, 0.3, 0.4)      # slide "Kaggle Metric"
EQUAL_WEIGHTS = (1/3, 1/3, 1/3)      # pagina Overview
REFS = {"task1": (34.0, 0.975, 120.0),
        "task2": (30.0, 0.930, 400.0),
        "task3": (34.0, 0.975, 150.0)}
TASK2_MAX_BYTES = 5500               # cap duro: sopra, il punteggio del task e' dimezzato


def task_score(psnr_v, msssim_v, cr, task="task1", weights=None):
    p_ref, m_ref, cr_ref = REFS[task]
    w1, w2, w3 = weights or SCORE_WEIGHTS
    cr = float(np.clip(cr, 1.0, 5000.0))
    return w1 * psnr_v / p_ref + w2 * msssim_v / m_ref + w3 * np.log(cr) / np.log(cr_ref)


# --------------------------------------------------------------------------------------
# backend classico
# --------------------------------------------------------------------------------------
def _available_backend():
    try:
        import pillow_avif  # noqa: F401
        return "AVIF"
    except Exception:
        pass
    if "AVIF" in Image.registered_extensions().values():
        return "AVIF"
    from PIL import features
    return "WEBP" if features.check("webp") else "JPEG"


BACKEND = _available_backend()


def backend_bytes(img_u8, q, fmt=None):
    fmt = fmt or BACKEND
    buf = io.BytesIO()
    if fmt == "AVIF":
        opts = dict(quality=int(q), speed=6)
    elif fmt == "WEBP":
        opts = dict(quality=int(q), method=6)
    else:
        opts = dict(quality=int(q), optimize=True, subsampling="4:2:0")
    Image.fromarray(img_u8).save(buf, format=fmt, **opts)
    return buf.getvalue()


def backend_decode(data):
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def resize_u8(img_u8, w, h, up=False):
    if img_u8.shape[1] == w and img_u8.shape[0] == h:
        return img_u8
    f = Image.BICUBIC if up else Image.LANCZOS
    return np.asarray(Image.fromarray(img_u8).resize((int(w), int(h)), f), dtype=np.uint8)


# --------------------------------------------------------------------------------------
# formato del bitstream
#   > H  h0      altezza originale
#   > H  w0      larghezza originale
#   > H  hs      altezza effettivamente codificata
#   > H  ws      larghezza effettivamente codificata
#   > B  q       qualita' del backend, oppure 0 = MINIATURA GREZZA (vedi sotto)
#
# Modalita' miniatura grezza (q = 0): i byte che seguono l'header sono RGB non compressi di
# una miniatura hs x ws. Serve perche' AVIF/WebP hanno un overhead di container fisso di
# qualche centinaio di byte: sotto quella soglia NESSUNA qualita' entra nel budget, e il
# Task 2 assegna zero a un payload fuori budget. Con la miniatura grezza la dimensione e'
# esattamente hs*ws*3 + header, quindi si puo' centrare qualunque budget >= 12 byte.
# --------------------------------------------------------------------------------------
HDR = ">HHHHB"
HDR_LEN = struct.calcsize(HDR)


def pack_payload(h0, w0, hs, ws, q, data):
    return struct.pack(HDR, h0, w0, hs, ws, int(q)) + data


def unpack_payload(payload):
    h0, w0, hs, ws, q = struct.unpack(HDR, payload[:HDR_LEN])
    return h0, w0, hs, ws, q, payload[HDR_LEN:]


def compress_raw(image_u8, q, scale=1.0, fmt=None):
    """Ritorna (payload, immagine_degradata_alla_dimensione_originale)."""
    h0, w0 = image_u8.shape[:2]
    if scale != 1.0:
        hs, ws = max(16, round(h0 * scale)), max(16, round(w0 * scale))
        src = resize_u8(image_u8, ws, hs)
    else:
        hs, ws, src = h0, w0, image_u8
    data = backend_bytes(src, q, fmt)
    return pack_payload(h0, w0, hs, ws, q, data), data


def raw_thumb_payload(image_u8, budget_bytes):
    """Miniatura RGB grezza che sta ESATTAMENTE nel budget dato. Ultima risorsa quando
    nemmeno la qualita' minima del backend ci entra."""
    h0, w0 = image_u8.shape[:2]
    n_pix = max(1, (int(budget_bytes) - HDR_LEN) // 3)
    ar = w0 / max(h0, 1)
    hs = max(1, int(np.sqrt(n_pix / max(ar, 1e-6))))
    ws = max(1, min(int(n_pix // hs), n_pix))
    while hs * ws > n_pix and hs > 1:
        hs -= 1
    thumb = resize_u8(image_u8, ws, hs)
    return pack_payload(h0, w0, hs, ws, 0, thumb.tobytes())


def decompress_raw(payload):
    """Decodifica SENZA restauro: e' l'input della rete di restauro."""
    h0, w0, hs, ws, q, data = unpack_payload(payload)
    if q == 0:                                   # miniatura grezza
        need = hs * ws * 3
        arr = np.frombuffer(data[:need], dtype=np.uint8)
        if arr.size < need:                      # difensivo: non dovrebbe accadere
            arr = np.pad(arr, (0, need - arr.size), mode="edge" if arr.size else "constant")
        dec = arr.reshape(hs, ws, 3)
    else:
        dec = backend_decode(data)
    if dec.shape[:2] != (h0, w0):
        dec = resize_u8(dec, w0, h0, up=True)
    return dec, q, (hs, ws), (h0, w0)


def degrade(image_u8, q, scale=1.0, fmt=None):
    """La pipeline della competizione senza la rete: serve a generare i dati di training
    del restauratore (input degradato -> target pulito)."""
    payload, _ = compress_raw(image_u8, q, scale, fmt)
    dec, _, _, _ = decompress_raw(payload)
    return dec


# ======================================================================================
# La rete di restauro (lato decoder)
#
# U-Net multi-scala condizionata su (q, scale): deve saper ripulire l'uscita del backend
# a MOLTI punti operativi diversi, non a uno solo — Task 2 lavora a bitrate molto piu'
# bassi di Task 1. Il condizionamento avviene per FiLM (una scala e uno shift per canale
# predetti dai due scalari), che e' il modo piu' economico per dare alla rete l'informazione
# "quanto e' degradato questo input".
#
# Accetta qualunque H,W e restituisce ESATTAMENTE la stessa forma (padding interno e crop),
# cosi' non puo' fallire il controllo di shape dell'harness.
# ======================================================================================
def _torch():
    import torch
    return torch


class FiLM:
    pass


def build_restorer(base=32, blocks_per_stage=2):
    """Costruito dentro una funzione cosi' hybrid_codec.py resta importabile senza torch
    (utile per testare formato del bitstream e logica di budget in ambienti senza GPU)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FiLMLayer(nn.Module):
        def __init__(self, ch, cond_dim=2):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(cond_dim, 32), nn.SiLU(), nn.Linear(32, ch * 2))
        def forward(self, x, cond):
            gb = self.net(cond)
            g, b = gb.chunk(2, dim=1)
            return x * (1 + g[:, :, None, None]) + b[:, :, None, None]

    class ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
            self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
            self.n1 = nn.GroupNorm(min(8, ch), ch)
            self.n2 = nn.GroupNorm(min(8, ch), ch)
            self.film = FiLMLayer(ch)
        def forward(self, x, cond):
            h = F.silu(self.n1(self.c1(x)))
            h = self.n2(self.c2(h))
            h = self.film(h, cond)
            return F.silu(x + h)

    class Stage(nn.Module):
        def __init__(self, ch, n):
            super().__init__()
            self.blocks = nn.ModuleList([ResBlock(ch) for _ in range(n)])
        def forward(self, x, cond):
            for b in self.blocks:
                x = b(x, cond)
            return x

    class Restorer(nn.Module):
        """3 livelli di scala. Predice un RESIDUO sull'input degradato: partire
        dall'identita' rende il training molto piu' stabile e veloce, perche' la rete deve
        imparare solo la correzione, non ricostruire l'immagine da zero."""
        MULT = 8      # padding a multiplo di 8 (2 downsample)

        def __init__(self, base=32, n=2):
            super().__init__()
            self.inc = nn.Conv2d(3, base, 3, 1, 1)
            self.s1 = Stage(base, n)
            self.d1 = nn.Conv2d(base, base * 2, 3, 2, 1)
            self.s2 = Stage(base * 2, n)
            self.d2 = nn.Conv2d(base * 2, base * 4, 3, 2, 1)
            self.s3 = Stage(base * 4, n)
            self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1)
            self.s2b = Stage(base * 2, n)
            self.u1 = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
            self.s1b = Stage(base, n)
            self.out = nn.Conv2d(base, 3, 3, 1, 1)
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

        def forward(self, x, cond):
            h0, w0 = x.shape[-2:]
            ph = (self.MULT - h0 % self.MULT) % self.MULT
            pw = (self.MULT - w0 % self.MULT) % self.MULT
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="reflect")
            inp = x
            e1 = self.s1(self.inc(x), cond)
            e2 = self.s2(self.d1(e1), cond)
            e3 = self.s3(self.d2(e2), cond)
            d2 = self.s2b(self.u2(e3) + e2, cond)
            d1 = self.s1b(self.u1(d2) + e1, cond)
            res = self.out(d1)
            y = (inp + res).clamp(-1, 1)
            return y[..., :h0, :w0]

    return Restorer(base=base, n=blocks_per_stage)


def make_cond(q, scale, device, dtype=None):
    import torch
    t = torch.tensor([[float(q) / 100.0, float(scale)]], device=device,
                     dtype=dtype or torch.float32)
    return t


# ======================================================================================
# I tre codec
# ======================================================================================
TILE = 512          # inferenza a tasselli: a risoluzione nativa un frame puo' non stare
TILE_OVERLAP = 32   # in memoria GPU. Sovrapposizione per evitare cuciture visibili.


class _RestoreBase:
    def __init__(self, weights_path=None, device=None, base=32, blocks_per_stage=2,
                 use_restorer=True):
        # torch importato PIGRAMENTE: senza restauro (weights_path=None) la pipeline e' puro
        # backend classico e deve restare utilizzabile e testabile anche dove torch non c'e'.
        self.torch = None
        self.device = device or "cpu"
        self.net = None
        if use_restorer and weights_path:
            import torch
            self.torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            # Determinismo: l'harness richiede encode() byte-identico su chiamate ripetute.
            # L'encoder qui e' il backend classico (gia' deterministico), ma fissiamo
            # comunque i flag perche' il restauro entra nella ricostruzione inviata.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            ck = torch.load(weights_path, map_location=self.device)
            arch = ck.get("arch", {})
            self.net = build_restorer(base=int(arch.get("base", base)),
                                      blocks_per_stage=int(arch.get("blocks_per_stage",
                                                                    blocks_per_stage)))
            self.net.load_state_dict(ck["restorer"])
            self.net.to(self.device).eval()

    def _restore(self, img_u8, q, scale):
        """Applica il restauro a tasselli. Se non c'e' rete, ritorna l'input invariato —
        cosi' la pipeline funziona (a qualita' inferiore) anche senza checkpoint."""
        if self.net is None:
            return img_u8
        torch = self.torch
        h, w = img_u8.shape[:2]
        x = torch.from_numpy(img_u8).permute(2, 0, 1).unsqueeze(0).float()
        x = (x.to(self.device) / 127.5) - 1.0
        cond = make_cond(q, scale, self.device)
        out = torch.zeros_like(x)
        wsum = torch.zeros((1, 1, h, w), device=self.device)

        # BUG CORRETTO QUI. La versione precedente generava le posizioni dei tasselli con
        # range(0, h - TILE + 1, step): se (h - TILE) non e' multiplo di step, l'ultima
        # posizione NON arriva al bordo e una striscia in basso/a destra non veniva coperta
        # da nessun tassello. Quei pixel restavano a zero nell'accumulatore e, dopo la
        # de-normalizzazione, uscivano grigio medio. Su 1024x2048 (le immagini vere del
        # dataset) erano il 7.7% dell'immagine; su 1080x1920 il 29.6%; su 700x900 il 58%.
        # Effetto: la PSNR veniva INCHIODATA dal contributo delle bande grigie e non
        # dipendeva piu' dalla qualita' del codec — nello sweep restava ~24.5 dB mentre il
        # CR variava di 76 volte. Su un ritaglio 512x512 (= TILE) il bug era invisibile,
        # perche' li' la copertura e' esatta.
        # Ora l'ultima posizione e' forzata al bordo, cosi' la copertura e' totale per
        # qualunque dimensione; l'assert sotto lo verifica a ogni chiamata.
        step = TILE - TILE_OVERLAP

        def _positions(size):
            last = max(size - TILE, 0)
            pos = list(range(0, last + 1, step))
            if pos[-1] != last:
                pos.append(last)
            return pos

        with torch.no_grad():
            for y0 in _positions(h):
                for x0 in _positions(w):
                    y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
                    y0c, x0c = max(0, y1 - TILE), max(0, x1 - TILE)
                    tile = x[..., y0c:y1, x0c:x1]
                    out[..., y0c:y1, x0c:x1] += self.net(tile, cond)
                    wsum[..., y0c:y1, x0c:x1] += 1.0
        assert bool((wsum > 0).all()), (
            f"copertura dei tasselli incompleta su {h}x{w}: e' il bug delle bande grigie, "
            "non lasciarlo passare in silenzio")
        out = out / wsum.clamp(min=1.0)
        out = ((out.clamp(-1, 1) + 1.0) * 127.5).round()
        return out.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()

    def _decode_payload(self, payload):
        dec, q, (hs, ws), (h0, w0) = decompress_raw(payload)
        scale = ws / max(w0, 1)
        return self._restore(dec, q, scale)


class HybridCodec(_RestoreBase):
    """Task 1 — punto operativo (scale, q) fissato alla costruzione, scelto misurando S."""
    name = "wsense-hybrid-codec"

    def __init__(self, weights_path=None, device=None, quality=50, scale=0.5, **kw):
        super().__init__(weights_path, device, **kw)
        self.quality, self.scale = int(quality), float(scale)

    def encode(self, image: np.ndarray) -> bytes:
        payload, _ = compress_raw(image, self.quality, self.scale)
        return payload

    def decode(self, payload: bytes) -> np.ndarray:
        return self._decode_payload(payload)


class HybridBudgetCodec(_RestoreBase):
    """Task 2 — deve stare sotto min(budget della riga, TASK2_MAX_BYTES).

    Cerca fra le scale dalla piu' grande alla piu' piccola e, per ciascuna, la qualita'
    piu' alta che ci sta (ricerca binaria). Tiene il risultato piu' grande fra quelli che
    entrano: a parita' di budget, piu' byte usati = piu' informazione trasmessa. Il
    fallback finale (scala minima, qualita' minima) sta in qualunque budget ragionevole.
    """
    name = "wsense-hybrid-budget-codec"
    SCALES = (0.60, 0.45, 0.35, 0.25, 0.18, 0.125, 0.09, 0.0625)

    def _fit(self, image, scale, room):
        lo, hi, best = 1, 92, None
        while lo <= hi:
            q = (lo + hi) // 2
            payload, data = compress_raw(image, q, scale)
            if len(payload) <= room + HDR_LEN:
                best, lo = payload, q + 1
            else:
                hi = q - 1
        return best

    def encode_at(self, image: np.ndarray, budget_bytes: int) -> bytes:
        budget = min(int(budget_bytes), TASK2_MAX_BYTES)
        best = None
        for s in self.SCALES:
            cand = self._fit(image, s, budget - HDR_LEN)
            if cand is not None and (best is None or len(cand) > len(best)):
                best = cand
                break     # la prima scala che entra e' anche la piu' informativa
        if best is None:
            # Nemmeno la scala/qualita' minima entra: succede su budget molto piccoli,
            # dove l'overhead di container di AVIF/WebP domina. La miniatura grezza sta
            # per costruzione dentro qualunque budget.
            best = raw_thumb_payload(image, budget)
        if len(best) > budget:                   # invariante del contratto del Task 2
            best = raw_thumb_payload(image, budget)
        return best

    def decode(self, payload: bytes) -> np.ndarray:
        return self._decode_payload(payload)


class HybridGroupCodec(_RestoreBase):
    """Task 3 — il CR e' calcolato sull'INTERA scena, quindi ogni immagine contribuisce
    allo stesso rapporto: si usa lo stesso punto operativo per tutte. Le immagini della
    stessa scena sono molto simili, ma il backend classico non ha predizione fra immagini;
    il guadagno qui viene dal punto operativo, scelto separatamente da quello del Task 1
    perche' CR_ref e' diverso (150 invece di 120)."""
    name = "wsense-hybrid-group-codec"

    def __init__(self, weights_path=None, device=None, quality=50, scale=0.5, **kw):
        super().__init__(weights_path, device, **kw)
        self.quality, self.scale = int(quality), float(scale)

    def encode_group(self, images: list) -> list:
        return [compress_raw(im, self.quality, self.scale)[0] for im in images]

    def decode_group(self, payloads: list) -> list:
        return [self._decode_payload(p) for p in payloads]


# ======================================================================================
# Scelta del punto operativo — per misura, non a intuito
# ======================================================================================
def sweep_task1(image_paths, scales=(1.0, 0.7, 0.5, 0.35, 0.25), qualities=(20, 35, 50, 65, 80),
                weights_path=None, device=None, task="task1", verbose=True, **kw):
    from challenge_common import load_image, psnr, ms_ssim, image_bytes
    imgs = [load_image(p) for p in image_paths]
    rows = []
    for s in scales:
        for q in qualities:
            c = HybridCodec(weights_path, device, quality=q, scale=s, **kw)
            crs, ps, ms = [], [], []
            for im in imgs:
                pay = c.encode(im)
                rec = c.decode(pay)
                crs.append(image_bytes(im) / max(len(pay), 1))
                ps.append(psnr(im, rec)); ms.append(ms_ssim(im, rec))
            cr, P, M = float(np.mean(crs)), float(np.mean(ps)), float(np.mean(ms))
            rows.append(dict(scale=s, quality=q, cr=cr, psnr=P, ms_ssim=M,
                             S=task_score(P, M, cr, task),
                             S_equal=task_score(P, M, cr, task, EQUAL_WEIGHTS)))
            if verbose:
                print(f"  scale {s:4.2f} q {q:3d} | CR {cr:8.1f} | PSNR {P:6.2f} "
                      f"| MS-SSIM {M:.4f} | S={rows[-1]['S']:.4f} "
                      f"(pesi uguali {rows[-1]['S_equal']:.4f})")
            del c
    rows.sort(key=lambda r: -r["S"])
    if verbose:
        b = rows[0]
        print(f"  -> migliore: scale={b['scale']} q={b['quality']}  S={b['S']:.4f}")
        be = max(rows, key=lambda r: r["S_equal"])
        if (be["scale"], be["quality"]) != (b["scale"], b["quality"]):
            print(f"  -> a pesi uguali sarebbe scale={be['scale']} q={be['quality']} "
                  f"(S={be['S_equal']:.4f}); le due letture della metrica non concordano")
    return rows


def sweep_task3(groups, scales=(1.0, 0.7, 0.5, 0.35, 0.25), qualities=(20, 35, 50, 65, 80),
                weights_path=None, device=None, verbose=True, **kw):
    from challenge_common import load_image, psnr, ms_ssim, image_bytes
    rows = []
    for s in scales:
        for q in qualities:
            c = HybridGroupCodec(weights_path, device, quality=q, scale=s, **kw)
            crs, ps, ms = [], [], []
            for _sc, paths in groups.items():
                imgs = [load_image(p) for p in paths]
                pays = [bytes(x) for x in c.encode_group(imgs)]
                recs = c.decode_group(pays)
                crs.append(sum(image_bytes(i) for i in imgs) / max(sum(len(p) for p in pays), 1))
                ps.append(float(np.mean([psnr(a, b) for a, b in zip(imgs, recs)])))
                ms.append(float(np.mean([ms_ssim(a, b) for a, b in zip(imgs, recs)])))
            cr, P, M = float(np.mean(crs)), float(np.mean(ps)), float(np.mean(ms))
            rows.append(dict(scale=s, quality=q, cr=cr, psnr=P, ms_ssim=M,
                             S=task_score(P, M, cr, "task3"),
                             S_equal=task_score(P, M, cr, "task3", EQUAL_WEIGHTS)))
            if verbose:
                print(f"  scale {s:4.2f} q {q:3d} | CR {cr:8.1f} | PSNR {P:6.2f} "
                      f"| MS-SSIM {M:.4f} | S={rows[-1]['S']:.4f}")
            del c
    rows.sort(key=lambda r: -r["S"])
    if verbose:
        b = rows[0]
        print(f"  -> migliore: scale={b['scale']} q={b['quality']}  S={b['S']:.4f}")
    return rows


# ======================================================================================
# Sweep VELOCE
#
# Lo sweep "pieno" (sweep_task1/sweep_task3) ricostruisce ogni immagine a risoluzione
# nativa per ogni punto operativo: con 25 combinazioni diventa proibitivo, perche' sia il
# restauro a tasselli sia la MS-SSIM di scipy costano proporzionalmente ai megapixel.
#
# Qui il costo si taglia separando le due misure:
#   - il RATE resta ESATTO: si codifica l'immagine intera e si contano i byte veri (la
#     codifica AVIF e' la parte economica);
#   - la QUALITA' si stima su pochi ritagli deterministici a piena risoluzione, dove
#     girano restauro e metriche.
# Il costo passa da "megapixel" a "PROBE_N ritagli da PROBE_CROP", tipicamente 10-20x piu'
# veloce, con una stima della qualita' sufficientemente fedele per ORDINARE i punti
# operativi — che e' tutto quello che serve per sceglierne uno.
# ======================================================================================
PROBE_CROP = 384
PROBE_N = 3


def _probe_boxes(h, w, size=PROBE_CROP, n=PROBE_N):
    if h <= size and w <= size:
        return [(0, 0, h, w)]
    ys = np.linspace(0, max(0, h - size), n).round().astype(int)
    xs = np.linspace(0, max(0, w - size), n).round().astype(int)
    hh, ww = min(h, size), min(w, size)
    return [(int(y), int(x), int(y) + hh, int(x) + ww) for y, x in zip(ys, xs)]


def sweep_fast(image_paths, scales=(1.0, 0.7, 0.5, 0.35, 0.25),
               qualities=(20, 35, 50, 65, 80), weights_path=None, device=None,
               task="task1", groups=None, verbose=True, **kw):
    """Sweep veloce. `groups` (dict scena -> percorsi) attiva la modalita' Task 3, in cui
    il CR e' calcolato sull'intera scena come fa il leaderboard."""
    from challenge_common import load_image, psnr, ms_ssim, image_bytes

    if groups is not None:
        items = [[load_image(p) for p in paths] for paths in groups.values()]
    else:
        items = [[load_image(p)] for p in image_paths]

    codec_cache = {}
    rows = []
    for s in scales:
        for q in qualities:
            key = (s, q)
            if key not in codec_cache:
                codec_cache[key] = HybridCodec(weights_path, device, quality=q, scale=s, **kw)
            c = codec_cache[key]
            crs, ps, ms = [], [], []
            for group in items:
                raw = sum(image_bytes(im) for im in group)
                total = 0
                for im in group:
                    total += len(c.encode(im))          # rate ESATTO, immagine intera
                crs.append(raw / max(total, 1))
                for im in group:                         # qualita' stimata sui ritagli
                    h, w = im.shape[:2]
                    for (y0, x0, y1, x1) in _probe_boxes(h, w):
                        crop = np.ascontiguousarray(im[y0:y1, x0:x1])
                        rec = c.decode(c.encode(crop))
                        ps.append(psnr(crop, rec)); ms.append(ms_ssim(crop, rec))
            cr, P, M = float(np.mean(crs)), float(np.mean(ps)), float(np.mean(ms))
            rows.append(dict(scale=s, quality=q, cr=cr, psnr=P, ms_ssim=M,
                             S=task_score(P, M, cr, task),
                             S_equal=task_score(P, M, cr, task, EQUAL_WEIGHTS)))
            if verbose:
                print(f"  scale {s:4.2f} q {q:3d} | CR {cr:8.1f} | PSNR {P:6.2f} "
                      f"| MS-SSIM {M:.4f} | S={rows[-1]['S']:.4f} "
                      f"(pesi uguali {rows[-1]['S_equal']:.4f})")
    rows.sort(key=lambda r: -r["S"])
    if verbose:
        b = rows[0]
        print(f"  -> migliore: scale={b['scale']} q={b['quality']}  S={b['S']:.4f}")
        be = max(rows, key=lambda r: r["S_equal"])
        if (be["scale"], be["quality"]) != (b["scale"], b["quality"]):
            print(f"  -> a pesi uguali sarebbe scale={be['scale']} q={be['quality']} "
                  f"(S={be['S_equal']:.4f}): le due letture della metrica non concordano")
    return rows
