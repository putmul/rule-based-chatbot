# SiagaBot

Chatbot rule-based sederhana untuk panduan pertolongan pertama cedera sehari-hari.

## Menyiapkan MariaDB XAMPP

1. Jalankan **MySQL/MariaDB** dari XAMPP.
2. Buka phpMyAdmin.
3. Import file `sql/schema.sql`, atau jalankan isinya pada SQL console.
4. Salin `.env.example` menjadi `.env` jika konfigurasi database berbeda.

Konfigurasi default:

```env
DB_HOST=127.0.0.1
DB_PORT=3306
DB_NAME=siagabot
DB_USER=root
DB_PASSWORD=
```

## Instalasi dependency

```bash
python -m pip install -r requirements.txt
```

## Menjalankan CLI

```bash
python -m chatbot.cli
```

Ketik `keluar`, `exit`, atau `quit` untuk mengakhiri sesi.

## Menjalankan web

Install dependency lalu jalankan server Flask:

```bash
python -m pip install -r requirements.txt
python app.py
```

Buka `http://127.0.0.1:5000/` untuk panel chatbot. Halaman statistik tersedia
di `http://127.0.0.1:5000/statistics` dan membutuhkan MariaDB agar histori chat
dapat ditampilkan.

Jika MariaDB belum aktif atau database belum dibuat, chatbot tetap dapat
menjawab pertanyaan, tetapi histori chat tidak akan disimpan.

## Menjalankan test

```bash
python -m unittest discover -s tests -v
```

Core chatbot berada di `chatbot/chatbot.py` dalam bentuk fungsi dan dapat
digunakan kembali oleh interface web pada tahap berikutnya.
