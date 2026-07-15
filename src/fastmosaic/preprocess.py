from __future__ import annotations

from pathlib import Path
import shutil
import cv2

# Gerekli veri yapılarını ve veri seti okuma fonksiyonlarını içe aktar
from fastmosaic.io.dataset import load_dataset, FrameMeta

# Bu içe aktarmalar projeye src/fastmosaic/calib/undistort.py dosyası eklendikten sonra çalışacaktır.
# Kamera kalibrasyon dosyasını okumak ve lens bozulmalarını gidermek için kullanılır.
from fastmosaic.calib.undistort import load_camera_yaml, undistort_image


def preprocess_dataset(
        raw_dir: str | Path,
        processed_dir: str | Path,
        camera_yaml: str | Path | None = "configs/camera.yaml",
) -> Path:
    """
    Ham veri setini işleyerek yeni bir işlenmiş veri seti oluşturur:
      - raw_dir konumundan verileri (frames/ klasörü ve varsa telemetry.csv dosyası) okur.
      - Eğer geçerli bir camera_yaml dosyası belirtilmişse, görüntülerdeki lens bozulmalarını (distortion) düzeltir.
      - İşlenmiş (düzeltilmiş) görüntü karelerini processed_dir/frames klasörüne yazar.
      - Ham veri setinde telemetry.csv dosyası mevcutsa, bu dosyayı da işlenmiş dizine kopyalar.
    """

    # Girdi ve çıktı dizin yollarını (string olarak verilmiş olsalar bile)
    # daha kolay dosya işlemleri yapabilmek için Path nesnelerine dönüştür
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    # İşlenmiş görüntülerin kaydedileceği 'frames' alt klasörünün yolunu belirle
    out_frames_dir = processed_dir / "frames"

    # Çıktı klasörünü (ve gerekirse eksik olan üst klasörlerini) oluştur,
    # klasör zaten mevcutsa hata fırlatmasını engelle (exist_ok=True)
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    # Ham veri setinden kare (frame) listesini (meta verilerini) ve eğer destekleniyorsa telemetri verilerini yükle
    metas = load_dataset(raw_dir)

    # Eğer kamera yapılandırma dosyası (yaml) belirtilmişse kamera kalibrasyon parametrelerini yükle,
    # belirtilmemişse veya dosya yoksa bu değişkeni None olarak bırak
    cam = load_camera_yaml(camera_yaml) if camera_yaml else None

    # Meta verilerindeki (metas) her bir kare (frame) bilgisi üzerinde döngü başlat
    for m in metas:
        # Görüntüyü diskten belleğe oku
        img = cv2.imread(str(m.path))

        # Görüntü dosyası bozuksa veya okunamadıysa bu kareyi atla ve sonrakine geç
        if img is None:
            continue

        # Eğer kamera kalibrasyon bilgisi başarıyla yüklendiyse,
        # görüntüdeki radyal veya teğetsel lens bozulmalarını (örneğin balık gözü etkisi) gider
        if cam is not None:
            img = undistort_image(img, cam)

        # Çıktı dosyası için karenin ID'sini (frame_id) kullanarak standart bir dosya adı oluştur
        out_name = f"frame_{m.frame_id}.jpg"

        # İşlenmiş ve düzeltilmiş görüntüyü yeni oluşturduğumuz çıktı klasörüne kaydet
        cv2.imwrite(str(out_frames_dir / out_name), img)

    # Orijinal (ham) veri setinde telemetri (örneğin GPS veya drone uçuş verileri) dosyası var mı kontrol et
    tele_in = raw_dir / "telemetry.csv"
    if tele_in.exists():
        # Eğer varsa, telemetri dosyasını meta verilerini (dosya izinleri, oluşturulma tarihi vb.) koruyarak
        # yeni işlenmiş veri seti klasörüne kopyala
        shutil.copy2(tele_in, processed_dir / "telemetry.csv")

    # İşlem tamamlandıktan sonra, ileride kullanıla bilmesi için oluşturulan yeni veri setinin kök dizin yolunu döndür
    return processed_dir