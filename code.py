from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# Parsing argument
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# Rentang warna hijau dalam HSV
greenLower = (29, 86, 6)
greenUpper = (64,255, 255)

# Membuat deque untuk menyimpan titik lintasan objek
pts = deque(maxlen=args["buffer"])

# Menginisialisasi video stream
if args["video"] is None:
    vs = VideoStream(src=0).start()
    print("INFO: Kamera diaktifkan...")
    time.sleep(2.0)  # Memberi waktu kamera untuk menyala
else:
    vs = cv2.VideoCapture(args["video"])
    print(f"INFO: Membuka video file: {args['video']}")

# Loop utama
while True:
    # Ambil frame
    frame = vs.read()
    if args["video"] is not None:
        frame = frame[1]  # Ambil frame kedua jika menggunakan VideoCapture

    # Periksa apakah frame kosong
    if frame is None:
        print("INFO: Tidak ada frame lagi, keluar...")
        break

    # Ubah ukuran frame agar lebih kecil untuk efisiensi
    frame = imutils.resize(frame, width=769, height=1920)

    # Konversi frame ke HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Buat mask untuk warna hijau
    mask = cv2.inRange(hsv, greenLower, greenUpper)

    # Membersihkan mask (menghapus noise)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Temukan kontur pada mask
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    center = None

    # Hanya lanjutkan jika ada setidaknya satu kontur yang ditemukan
    if len(contours) > 0:
        # Temukan kontur terbesar dan hitung lingkaran yang mengelilinginya
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        # Hitung pusat lingkaran
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Lanjutkan hanya jika radius lebih besar dari threshold tertentu
        if radius > 10:
            # Gambar lingkaran di sekitar objek
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

            # Gambar titik pusat
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # Tambahkan pusat ke deque untuk pelacakan lintasan
    pts.appendleft(center)

    # Gambar lintasan objek
    for i in range(1, len(pts)):
        # Lewati jika salah satu titik adalah None
        if pts[i - 1] is None or pts[i] is None:
            continue

        # Hitung ketebalan garis berdasarkan jarak
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # Tampilkan frame hasil
    cv2.imshow("Kamera", frame)

    # Keluar jika pengguna menekan tombol 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord("e"):
        print("INFO: Tombol 'E'(Exit) ditekan, keluar...")
        break

# Hentikan video stream atau release video
if args["video"] is None:
    vs.stop()
else:
    vs.release()

# Tutup semua jendela
cv2.destroyAllWindows()
print("INFO: Program selesai.")
