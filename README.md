## Tubes AI
### Anggota Kelompok:
| Nama | NIM |
| --- | --- |
| Muhammad Nur Majiid|13524028|
| Stefani Angeline Oroh|13524064|
| Angelina Andra Alanna|13524079|

### Setup
```
pip install -r requirements.txt
```

### Menjalankan program
```
cd src
python main.py
```
Pilih dataset (nomor dari daftar `src/data/` atau path file) dan objective function, lalu pilih menu:

| Menu | Fungsi |
|---|---|
| 1–3 | Satu run SA, GA, atau HC (varian dipilih) dengan seed tertentu |
| 4 | Eksperimen untuk laporan dengan semua seed di `experiments.json`: HC 4 varian, SA, GA eksperimen A & B |
| 5 | Membuat file replay untuk viewer 3D |
| 6 | Membuat ulang plot untuk semua hasil dataset yang dipilih |
| 7 | Ganti dataset / objective |

Hasil setiap run disimpan di `output/<algoritma>/<dataset>/...` (`initial_state.json`, `final_state.json`, `history.csv`, `summary.json`), beserta plotnya:
- `objective.png`: objective terhadap iterasi (GA: objective maksimum dan rata-rata populasi)
- `accept_prob.png`: e^(ΔE/T) terhadap iterasi (khusus Simulated Annealing)
- `initial_state.png`, `final_state.png`: susunan paket 3D per truk, beserta daftar paket di luar truk

### Visualisasi
Membuat ulang plot untuk hasil yang sudah ada (dari folder `src/`):
```
python -m visualisasi.plots ../output                 # semua run
python -m visualisasi.plots ../output/sa/test_case    # sebagian
```

Viewer 3D interaktif: buka `src/visualisasi/viewer3d.html` di browser (butuh internet untuk memuat Three.js).
Viewer bisa memuat dataset, file state (`initial_state.json`/`final_state.json`, muat dataset-nya dulu), atau file replay.
Membuat file replay (state tiap langkah algoritma) dari folder `src/`:
```
python -m visualisasi.export_replay multi_truck_case.json --algo sa --seed 1
python -m visualisasi.export_replay --algo ga --pop 50 --iter 100
python -m visualisasi.export_replay --algo hc --variant sideways
```
Hasilnya ada di `output/replay/`.
