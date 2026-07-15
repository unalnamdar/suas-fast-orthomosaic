from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from fastmosaic.io.dataset import FrameMeta


@dataclass
class StitchConfig:
    """
    Görüntü birleştirme (stitching) işlemi için yapılandırma parametrelerini tutan veri sınıfı.
    """
    # Çıkarılacak maksimum ORB feature sayısı (fotoğraf başına 3000 tane common point seçilebilir)
    orb_features: int = 3000

    # Lowe'nin oran testi (ratio test) için eşik değeri (birinci eşleşme ikinci eşleşmeden %25 daha iyi olursa eşleşmeyi tut
    # Aksi olursa ele; Bu da tahmin edileceği üzere eşleşme doğruluğunu arttırır.
    ratio_test: float = 0.75

    # --- Çok büyük canvas ve bellek sorunlarından kaçınmak için güvenlik sınırları ---

    # Her bir girdi görüntüsü bu boyuttan (genişlik veya yükseklik) büyükse yeniden boyutlandırılır
    max_input_dim: int = 1200
    # Canvas bu boyutu aşarsa otomatik olarak küçültülür (downscale)
    max_canvas_dim: int = 16000
    # Canvas bu hard limit'i aşacak bir boyuta ulaşırsa, o kare güvenli tutmak için atlanır
    hard_canvas_dim: int = 12000

    # --- RANSAC (aykırı değerleri eleme) ve kalite ayarları ---
    # RANSAC (Random Sample Consensus) kısaca ORB ve Lowe abimizin testlerinden geçen ama hatalı eşleşmeleri ayıklayan bi algoritma.

    # RANSAC algoritması için piksel cinsinden hata toleransı (reprojection threshold)
    affine_ransac_thresh_px: float = 3.0
    # Geçerli bir eşleşme sayılabilmesi için gereken minimum içte kalan (inlier) nokta sayısı
    min_inliers: int = 25


def _match_orb(img_prev, img_cur, cfg: StitchConfig):
    """
    İki ardışık görüntü (önceki ve mevcut) arasındaki ORB özelliklerini bulur ve eşleştirir.
    Eşleşme yeterli değilse None döner, yeterliyse her iki görüntüdeki eşleşen noktaların koordinatlarını döner.
    """
    # Belirtilen özellik sayısı ile ORB (Oriented FAST and Rotated BRIEF) nesnesini oluştur
    orb = cv2.ORB_create(nfeatures=cfg.orb_features)

    # Önceki ve mevcut görüntü için anahtar noktaları (keypoints) ve tanımlayıcıları (descriptors) tespit et
    k1, d1 = orb.detectAndCompute(img_prev, None)
    k2, d2 = orb.detectAndCompute(img_cur, None)

    # Tanımlayıcı bulunamadıysa veya anahtar nokta sayısı 12'den azsa işlemi iptal et
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None

    # Brute-Force (Kaba Kuvvet) eşleştirici oluştur (Hamming mesafesi kullanılarak)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # Her bir özellik için en iyi 2 eşleşmeyi (k=2) bul (KNN - K-Nearest Neighbors)
    pairs = bf.knnMatch(d1, d2, k=2)

    # Lowe'nin Oran Testi (Ratio Test) ile kaliteli eşleşmeleri filtrele
    good = []
    for m, n in pairs:
        # En iyi eşleşmenin mesafesi, ikinci en iyi eşleşmenin mesafesinin belirli bir oranından küçükse kabul et
        if m.distance < cfg.ratio_test * n.distance:
            good.append(m)

    # Geçerli eşleşme sayısı çok düşükse (12'den az) dönüşümü sağlıklı hesaplayamayacağı için iptal et
    if len(good) < 12:
        return None

    # Eşleşen anahtar noktaların piksel koordinatlarını (x, y) Numpy dizisi formatına dönüştür
    pts_prev = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_cur = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    return pts_prev, pts_cur


def stitch_sequence(metas: List[FrameMeta], out_path: str | Path, cfg: StitchConfig) -> None:
    """
    Verilen meta veri (FrameMeta) listesindeki görüntüleri sırasıyla okur,
    birbirine yapıştırır (stitch) ve nihai mozaiği belirtilen yola kaydeder.
    """
    # Çıktı yolunu hazırla ve gerekirse üst klasörleri oluştur
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _resize_if_needed(img):
        """
        Girdi görüntüsünün boyutları maksimum sınırı (max_input_dim) aşıyorsa
        görüntüyü orantılı olarak küçültür.
        """
        h, w = img.shape[:2]
        m = max(h, w)
        # Sınırın altındaysa orijinal görüntüyü geri dön
        if m <= cfg.max_input_dim:
            return img

        # Küçültme oranını hesapla ve yeni boyutları belirle
        s = cfg.max_input_dim / float(m)
        new_w = int(w * s)
        new_h = int(h * s)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Dizideki ilk (baz/temel) görüntüyü oku
    base = cv2.imread(str(metas[0].path))
    if base is None:
        raise RuntimeError(f"Okunamıyor: {metas[0].path}")

    # Gerekirse ilk görüntüyü boyutlandır
    base = _resize_if_needed(base)

    # Başlangıçta canvas ilk görüntünün kendisidir
    canvas = base
    # H_canvas şu anlama gelir: "Önceki başarıyla eklenen kareden → mevcut canvas koordinatlarına dönüşüm matrisi"
    # Başlangıçta canvas ve ilk görüntü aynı olduğu için bu birim (Identity) matristir.
    H_canvas = np.eye(3, dtype=np.float64)
    prev = base

    # Diğer kareleri sırayla dön ve canvas'a ekle
    for i in range(1, len(metas)):
        cur = cv2.imread(str(metas[i].path))
        if cur is None:
            continue
        cur = _resize_if_needed(cur)

        # Önceki kare ile mevcut kare arasındaki özellikleri eşleştir
        match = _match_orb(prev, cur, cfg)
        if match is None:
            # Eşleşme başarısızsa bu kareyi atla, ancak mevcut kareyi "önceki" olarak güncelle
            prev = cur
            continue
        pts_prev, pts_cur = match

        # --- Afin dönüşüm tahmini (Hızlı mozaiklemelerde homografi matrisine göre daha kararlıdır) ---
        # M dönüşüm matrisi mevcut kareyi (cur) önceki kareye (prev) dönüştürür
        M, inliers = cv2.estimateAffinePartial2D(
            pts_cur,
            pts_prev,
            method=cv2.RANSAC,
            ransacReprojThreshold=cfg.affine_ransac_thresh_px,
        )

        if M is None or inliers is None:
            prev = cur
            continue

        # İçte kalan (inlier) noktaların sayısını kontrol et, yetersizse atla
        inlier_count = int(inliers.sum())
        if inlier_count < cfg.min_inliers:
            prev = cur
            continue

        # 2x3 olan Afin matrisi homojen işlemler yapabilmek için 3x3 matrise çevir
        A = np.vstack([M, [0, 0, 1]])  # cur -> prev dönüşümü

        # Mevcut kareden (cur) güncel canvas'a olan genel dönüşümü hesapla
        # (cur -> prev) matrisi ile (prev -> canvas) matrisini çarp = H_canvas @ A
        H_cur_to_canvas = H_canvas @ A

        # --- Yeni canvas sınırlarını (boyutlarını) hesapla ---
        h1, w1 = canvas.shape[:2]
        h2, w2 = cur.shape[:2]

        # Mevcut karenin 4 köşesinin canvas üzerindeki yeni koordinatlarını bul
        corners_cur = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        warped_corners = cv2.transform(corners_cur, H_cur_to_canvas[:2])

        # Mevcut canvas'ın köşeleri (kendi lokal koordinat sisteminde)
        corners_canvas = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)

        # Canvas'ı genişletmek için eski canvas köşeleri ile yeni eklenecek karenin köşelerini birleştir
        all_pts = np.concatenate([corners_canvas, warped_corners], axis=0)

        # Yeni bileşik görüntünün minimum ve maksimum (x, y) sınırlarını bul
        xmin, ymin = np.floor(all_pts.min(axis=0).ravel()).astype(int)
        xmax, ymax = np.ceil(all_pts.max(axis=0).ravel()).astype(int)

        # Resmin negatif x veya y koordinatlarına taşmasını önlemek için gereken öteleme (translation) miktarı
        tx = -xmin if xmin < 0 else 0
        ty = -ymin if ymin < 0 else 0

        # Öteleme işlemi için dönüşüm matrisi (Translation Matrix)
        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
        new_w = int(xmax - xmin)
        new_h = int(ymax - ymin)

        # Kesin güvenlik kontrolü (Hard safety): Eğer hesaplanan yeni canvas boyutu donanımı zorlayacak
        # kadar büyük (hard_canvas_dim) veya hatalı (<=0) çıkarsa bu birleştirmeyi iptal et.
        if new_w > cfg.hard_canvas_dim or new_h > cfg.hard_canvas_dim or new_w <= 0 or new_h <= 0:
            prev = cur
            continue

        # Eski canvas'ı, negatif taşımları engellemek üzere yeni pozitif koordinat sistemine ötele
        new_canvas = cv2.warpAffine(canvas, T[:2], (new_w, new_h))

        # Mevcut kareyi de aynı yeni canvas koordinat sistemine çarpıtarak (warp) yerleştir
        # T matrisini uygulayarak canvas'taki ötelenmeyi hesaba kat
        H_cur_to_new_canvas = T @ H_cur_to_canvas
        cur_warp = cv2.warpAffine(cur, H_cur_to_new_canvas[:2], (new_w, new_h))

        # Basit üst üste bindirme (overlay) mantığı
        # Yeni gelen kareden boş (siyah) olmayan yerlerin maskesini çıkar
        mask_cur = (cv2.cvtColor(cur_warp, cv2.COLOR_BGR2GRAY) != 0)[:, :, None]
        # Maskenin olduğu yerlere yeni kareyi yaz, olmayan yerlerde eski canvas kalsın
        new_canvas = np.where(mask_cur, cur_warp, new_canvas).astype(np.uint8)
        canvas = new_canvas

        # Sonraki iterasyon için H_canvas'ı (Mevcut canvas matrisini) güncelle
        # Bir sonraki kare 'cur' ile eşleşecek ve YENİ canvas alanına dönüştürülmesi gerekecek.
        # Bu yüzden: H_canvas "cur -> new_canvas" dönüşümünü temsil etmelidir.
        H_canvas = H_cur_to_new_canvas

        # Canvas çok büyürse belleği korumak için canvas'ı ölçeklendir (küçült)
        # ve H_canvas matrisini buna uyacak şekilde güncelle
        ch, cw = canvas.shape[:2]
        cm = max(ch, cw)
        if cm > cfg.max_canvas_dim:
            s2 = cfg.max_canvas_dim / float(cm)
            new_cw = int(cw * s2)
            new_ch = int(ch * s2)
            canvas = cv2.resize(canvas, (new_cw, new_ch), interpolation=cv2.INTER_AREA)

            # Alt örneklemeyi (küçültmeyi) hesaba katmak için Ölçekleme (Scale) matrisi oluştur
            S = np.array([[s2, 0, 0], [0, s2, 0], [0, 0, 1]], dtype=np.float64)
            # Canvas'ı ölçeklendirdiğimiz için bir sonraki karelerin de aynı oranda
            # doğru yere yerleşmesi adına H_canvas matrisini ölçekle çarp
            H_canvas = S @ H_canvas

        # Döngünün bir sonraki adımı için mevcut kareyi 'önceki kare' olarak ata
        prev = cur

    # Nihai oluşan mozaik canvas'ını belirtilen dosya yoluna kaydet
    cv2.imwrite(str(out_path), canvas)

    def stitch_sequence(metas, out_path, cfg):
        print(f"Starting stitch: {len(metas)} frames")
        # ... mevcut kod devam ediyor