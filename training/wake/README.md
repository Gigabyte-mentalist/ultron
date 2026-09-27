# Wake word o'qitish muhiti (docs/tz/10_wake_word_training.md)

Bu papka **WSL2 Ubuntu ichida** ishlaydigan openWakeWord o'qitish pipeline'ini
joylaydi. Windows'dagi asosiy loyiha (`C:\ultron\`, uv/`.venv`) bunga
tegilmaydi -- faqat `data/models/wake/altron.onnx` natijasini iste'mol qiladi
(`config.yaml`'dagi `wake.oww_model_path`).

WSL'dan bu papka `/mnt/c/ultron/training/wake` sifatida ko'rinadi.

## 0. Muhit holati (bu mashinada allaqachon tekshirilgan)

- WSL2 + **Ubuntu-24.04.4 LTS** allaqachon o'rnatilgan (`wsl --list --verbose`).
  docs/tz/10 §3.1'dagi `wsl --install -d Ubuntu-24.04` qadami **kerak emas**.
- GPU passthrough ishlaydi: WSL ichida `nvidia-smi` RTX 4060'ni ko'rsatadi.
- WSL'ning o'z fayl tizimi (`/`, ext4) alohida, ~950 GB bo'sh joy bor --
  20-40 GB'lik sintetik ma'lumot uchun `C:\`dagi joy tanqisligi (01-bosqichda
  qayd etilgan) bu yerda muammo emas, hammasi WSL ichida (`~/`) saqlanadi,
  `/mnt/c/...`ga emas.
- Python 3.12.3 tayyor, lekin `openwakeword`/`piper` hali o'rnatilmagan.

**Claude Code bu qadamlarni bajarmaydi** -- WSL sizning umumiy dev muhitingiz
(`.aws`, `.azure`, `docker-lab` kabi boshqa loyihalar ham bor ekan), shuning
uchun `git clone`/`pip install`/katta datasetlarni yuklab olishni **o'zingiz**
ishga tushirasiz (`docs/tz/10_wake_word_training.md`ning o'z prompti ham aynan
shuni so'raydi: "WSL2 muhitini tayyorlash qadamlarini ber, men bajaraman").

## 1. O'qitish muhitini tayyorlash (WSL ichida, bir marta)

```bash
wsl -d Ubuntu
cd /mnt/c/ultron/training/wake
bash setup_env.sh          # ~10-20 daqiqa: klonlash, pip install, datasetlar
```

`setup_env.sh` nima qiladi (barchasi `~/oww_train/` ichida, loyihaning
`.venv`siga tegmaydi):
1. Python virtualenv (`~/oww_train/venv`).
2. `piper-sample-generator` (rhasspy fork'i) + `en_US-libritts_r-medium.pt`
   ovoz modeli -- "altron"/"ultron" so'zlari inglizcha fonetika bilan shu
   TTS orqali sintez qilinadi (docs/tz/10 §3.2's o'zi ham shuni ko'zda
   tutgan: "inglizcha fonetika bilan o'zbekcha talaffuzga yaqinlashtirish").
   **openWakeWord hozircha faqat inglizcha TTS'ni qo'llab-quvvatlaydi**
   (rasmiy README'ning "Language Support" bo'limi) -- bu TZ-10 farqi sifatida
   pastda §5'da qayd etilgan.
3. `openwakeword`ning git klonini editable o'rnatish (`pip install -e`) --
   PyPI'dagi versiyada `train.py`/notebook yo'q, faqat manba kodida bor.
4. Trening uchun qo'shimcha kutubxonalar (`speechbrain`, `audiomentations`,
   `torch-audiomentations`, `tensorflow-cpu` (tflite eksport uchun, ixtiyoriy),
   va h.k. -- to'liq ro'yxat rasmiy notebook'dan olingan).
5. Umumiy melspec/embedding ONNX modellarini klonlangan repo ichiga
   (`openwakeword/openwakeword/resources/models/`) yuklaydi -- xuddi Windows
   tomonidagi `scripts/download_models.py --oww` allaqachon qilgani kabi,
   **shu bitta rasmiy utility** (`openwakeword.utils.download_models`) orqali
   (notebook'ning eskirgan, muayyan versiyaga (`v0.5.1`) qadalgan `wget`
   havolalari o'rniga -- bu yerda ular ataylab ishlatilmadi).
6. Negativ feature dataset: `openwakeword_features_ACAV100M_2000_hrs_16bit.npy`
   (~2000 soat, HuggingFace `davidscripka/openwakeword_features`) va
   `validation_set_features.npy` (~11 soat, false-positive validatsiya uchun).
7. Fon shovqini uchun: MIT Room Impulse Response'lar (HF
   `davidscripka/MIT_environmental_impulse_responses`), AudioSet'ning bitta
   shard'i, FMA (Free Music Archive)'ning "small" bo'limidan bir soat --
   docs/tz/10 §3.2'ning "room impulse response, fon shovqini (musiqa, gap,
   ko'cha)" talabini qondiradi. To'liq ishlab chiqarish uchun bu datasetlarni
   kengaytirish tavsiya etiladi (rasmiy notebook o'zi ham shuni aytadi) --
   quyida "Kengaytirish" bo'limiga qarang.

Disk: ~15-25 GB (asosan ACAV100M feature fayli va AudioSet/FMA namunalar).

## 2. Haqiqiy namunalarni tayyorlash (Windows tomonida, mikrofon bilan)

Bular **fizik amallar** -- Claude Code mikrofonga ega emas, o'zingiz
bajarasiz (`docs/tz/04_wake_word.md`/`docs/tz/10...`ning o'z uslubi):

```powershell
cd C:\ultron
uv run python scripts/record_samples.py wake --count 202       # 98 ta bor edi (04-TZ) -> jami 300
uv run python scripts/record_samples.py wake-hardneg --reps 20  # "qiyin negativ"lar, 8 so'z x 20 = 160 ta
uv run python scripts/record_samples.py wake-neg --minutes 60   # kamida 2 soat uchun 2-3 marta ishga tushiring
```

Bundan tashqari, `config.yaml`'dagi `memory.save_utterances: true` bilan
(bu bosqichda yoqilgan) real foydalanish davomida har bir wake-trigger'ning
boshidagi 1.5 soniyasi avtomatik `data/recordings/wake_auto/`ga yig'iladi --
bu eng qimmatli manba (haqiqiy, tabiiy foydalanish sharoiti), lekin vaqt
talab qiladi (TZ: "kamida 1 hafta real foydalanishdan keyin").

Namunalar (Windows'da, `D:\ultron_data\recordings\` yoki `config.app.data_dir`
qayerga sozlangan bo'lsa) WSL'dan `/mnt/d/ultron_data/recordings/...` orqali
to'g'ridan-to'g'ri ko'rinadi -- alohida ko'chirish/nusxalash kerak emas.

## 3. Real namunalarni feature'ga aylantirish

```bash
wsl -d Ubuntu
cd ~/oww_train                      # not training/wake/ -- output must land where
source venv/bin/activate            # configs/altron_v2.yml's relative path expects it
python /mnt/c/ultron/training/wake/prepare_real_features.py \
    --recordings-dir /mnt/d/ultron_data/recordings \
    --output real_positive_features.npy
```

**Muhim ochiq masala (TZ-10 va rasmiy hujjat farqi, §5'da batafsil):**
openWakeWord'ning rasmiy notebook/hujjatlarida "o'z ovozli namunalaringizni
asosiy modelning trening ma'lumotiga (feature_data_files) qo'shish" hech
qachon ko'rsatilmagan -- faqat katta tayyor NEGATIV datasetlar (ACAV100M)
uchun ishlatilgan namuna bor. `prepare_real_features.py` kutubxonaning OMMAVIY
(lekin aynan shu maqsad uchun hujjatlashtirilmagan) funksiyalaridan
(`openwakeword.utils.compute_features_from_generator`,
`openwakeword.data.load_audio_clips`) tuzilgan -- texnik jihatdan to'g'ri
bo'lishi kerak (xuddi shu funksiyalar ACAV100M fayli tuzishda ishlatilgan),
lekin rasmiy ravishda sinovdan o'tkazilmagan yo'l. `altron_v1.onnx` (faqat
sintetik) har doim ishonchli zaxira variant sifatida qoladi.

## 4. O'qitish

```bash
bash run_training.sh configs/altron_v1.yml   # faqat sintetik, ~1-2 soat (GPU/CPU'ga qarab)
bash run_training.sh configs/altron_v2.yml   # sintetik + real (oversampled), 3-bosqichdan keyin
```

Har biri uchta bosqichni bajaradi (`--generate_clips`, `--augment_clips`,
`--train_model`) -- birinchisi uzilib qolsa, qayta ishga tushirish
xavfsiz (skript mavjud fayllar sonini tekshiradi, faqat yetishmaganini
generatsiya qiladi -- rasmiy notebook'ning o'zi ham shuni tasdiqlaydi).

Natija: `./altron_v1/altron.onnx`, `./altron_v2/altron.onnx`.

## 5. Windows'ga ko'chirish, benchmark, tanlash

```powershell
# ikkalasini ham benchmark qilib solishtirish uchun avval birini config'ga qo'yib ko'rish kerak;
# eng oddiy yo'l: v2'ni birinchi sinab ko'rish (real ma'lumot bilan, ko'proq va'da beradi)
Copy-Item \\wsl$\Ubuntu\home\<user>\... # yoki WSL ichidan: cp altron_v2/altron.onnx /mnt/c/ultron/data/models/wake/altron.onnx
cd C:\ultron
uv run python scripts/benchmark_wake.py --mode oww    # threshold tanlaydi, config.yaml'ga yozadi
# v1 bilan ham takrorlang (boshqa nomga saqlab, --positives/--negatives bilan solishtiring)
uv run python scripts/train_wake_verifier.py           # shaxsiy verifier (ixtiyoriy, §3.4)
uv run python scripts/benchmark_wake.py --mode all     # to'liq 10-TZ §3.5 jadvali, wake.mode'ni hal qiladi
```

## 6. openWakeWord rasmiy hujjati vs TZ-10: farqlar (Claude Code tekshiruvi, 2026-09-27)

TZ-10'ning o'z prompti (§5) buni birinchi qadam sifatida talab qiladi.

| TZ-10 aytgan | Rasmiy hujjat (dscripka/openWakeWord, `main` filiali) | Xulosa |
|---|---|---|
| "kamida 20 000 klip" | Config'dagi standart `n_samples: 10000`, izohda "minimum 20,000 tavsiya etiladi, ko'pincha 100,000+ yaxshiroq" | TZ to'g'ri, `configs/*.yml`da 20000 qo'yildi |
| "openWakeWord taqdim etadigan umumiy negativ feature to'plamlari" | Aniq mavjud: `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` (~2000 soat, HuggingFace) + validatsiya uchun 11 soatlik fayl | Mos keladi, §1'da yozilgan |
| "Piper TTS asosidagi sample generator" | To'g'ri, lekin **faqat inglizcha** ovoz modeli (`en_US-libritts_r-medium`) -- kutubxona hozircha boshqa tilni rasman qo'llab-quvvatlamaydi | TZ buni allaqachon hisobga olgan (inglizcha fonetika bilan yaqinlashtirish), ziddiyat yo'q, faqat aniqlashtirish |
| "real namunalar oversampling ×10" (asosiy modelga) | Rasmiy hujjatda **hech qanday misol yo'q** -- bu faqat "custom verifier" (ikkinchi, alohida logistik regressiya modeli) uchun hujjatlashtirilgan | **Haqiqiy farq.** §3'dagi kabi, ommaviy-lekin-hujjatlashtirilmagan funksiyalardan `prepare_real_features.py` tuzildi; rasmiy va sinab ko'rilgan yo'l -- shaxsiy verifier (§3.4, quyida) |
| Kutubxona versiyasi | Oxirgi tag: v0.6.0 (2024-02-11), 1.5+ yildan beri yangi relizsiz, lekin repo faol (commit/PR bor, arxivlanmagan) | TZ'ning o'zi "kutubxona versiyasiga qarab o'zgarishi mumkin, rasmiy hujjat asosiy manba" deb kutgan holat -- shunga amal qilindi |
| "piper-sample-generator" manbasi | TZ/kutubxona hujjatlarining ikkalasida ham (`dscripka/...` vs `rhasspy/...`) mos kelmaydigan fork nomi ko'rsatilgan | `setup_env.sh` notebook'ning **haqiqiy** `git clone` buyrug'idagi (`rhasspy/piper-sample-generator`) manbani ishlatadi |

## 7. Custom verifier (§3.4) -- rasmiy, sinab ko'rilgan yo'l

`scripts/train_wake_verifier.py` (Windows tomonida, `C:\ultron\.venv`da --
WSL/Piper shart emas, faqat `openwakeword`+`scikit-learn`, ikkalasi ham allaqachon
o'rnatilgan) `data/recordings/wake`+`wake_auto`dagi haqiqiy pozitiv va
`data/recordings/wake_neg`+`wake_hardneg`dagi negativlardan
`openwakeword.train_custom_verifier()` (logistik regressiya) chaqiradi va
natijani `data/models/wake/altron_verifier.pkl`ga yozadi, `config.yaml`'ning
`wake.oww_verifier_path`ini to'ldiradi. Rasmiy hujjat pozitiv uchun "3
tadan kam bo'lmasin, lekin ko'p ham bo'lmasin" deydi -- skript 300+ ta
mavjud pozitivdan tasodifiy ~40 tasini tanlaydi (`--max-positive` bilan
sozlanadi), hammasini emas.

**Ixtiyoriy:** oiladagi boshqa odamlar ham "Ultron" deb chaqirishi kerak
bo'lsa, `config.yaml`'da `wake.oww_verifier_path: null` qoldiring --
verifier faqat sizning ovozingizga moslashtirilgan, boshqalarni rad etadi.

## 8. Kengaytirish (ixtiyoriy, sifat uchun)

- `n_samples`ni 20000'dan 50000-100000'ga oshirish (disk/vaqt narxiga).
- `setup_env.sh`dagi AudioSet/FMA namunalarini kengaytirish (rasmiy notebook
  o'zi "to'liq production uchun butun datasetni yuklang" deydi).
- `background_paths`ga `data/recordings/wake_neg`ning o'zini ham qo'shish
  (`background_paths_duplication_rate` bilan oshirib) -- xonangizning haqiqiy
  akustikasini sintetik namunalarga ham singdirish uchun.
