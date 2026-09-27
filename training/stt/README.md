# STT LoRA fine-tune muhiti (docs/tz/11_uzbek_stt_finetune.md)

Bu papka **WSL2 Ubuntu ichida** ishlaydigan, shaxsiy ovoz bilan STT LoRA
fine-tune qiladigan pipeline'ni joylaydi. Windows'dagi asosiy loyiha
(`C:\ultron\`, uv/`.venv`) bunga tegilmaydi -- faqat yakuniy natijani
(`data/models/stt/ultron-uz-vN-ct2/`) iste'mol qiladi. Xuddi
`training/wake/`ga o'xshab (docs/tz/10), lekin bu safar alohida `venv`
(WSL'ning o'z Python'i bilan) yetarli -- openWakeWord kabi manba kodidan
editable o'rnatiladigan kutubxona yo'q, hammasi PyPI'da.

WSL'dan bu papka `/mnt/c/ultron/training/stt` sifatida ko'rinadi.

## 0. Nima uchun WSL/alohida venv (TZ §3.4)

`C:\ultron\.venv` — `faster-whisper`/`transformers`/`accelerate` bor, lekin
**`peft`/`datasets`/`bitsandbytes`/`evaluate` yo'q** (ataylab -- bular faqat
o'qitish uchun kerak, asosiy runtime'ga hech qachon kirmaydi). WSL2 +
Ubuntu-24.04.4 va GPU passthrough allaqachon `training/wake/`da tekshirilgan
va ishlagan (`training/wake/README.md` §0), shu muhitning o'zi qayta
ishlatiladi.

## 1. O'qitish muhitini tayyorlash (bir marta)

```bash
wsl -d Ubuntu
cd /mnt/c/ultron/training/stt
bash setup_env.sh          # ~10-20 daqiqa: torch (CUDA) + transformers/peft/datasets/...
```

## 2. Shaxsiy ma'lumot: yig'ish, belgilash, dataset qurish (Windows tomonida)

Bular Windows asosiy loyihada (`C:\ultron`, `uv run`) bajariladi, WSL shart
emas:

1. `config.yaml`'da `memory.save_utterances: true` (standart -- allaqachon
   yoqilgan) bo'lsa, har bir haqiqiy buyruq avtomatik
   `data/recordings/utterances/YYYY-MM-DD/`ga yoziladi (docs/tz/11 §3.1,
   `core/orchestrator.py` + `memory/utterance_log.py`). **Maqsad: 3-5 soat
   (birinchi tajriba uchun 1 soat ham yetarli)** -- kunlar davomida oddiy
   foydalanishning o'zi yetarli, alohida yozib olish shart emas.
2. `uv run python scripts/label_tool.py` -- brauzerda ochiladigan belgilash
   vositasi (127.0.0.1:8765). Past ishonchli va muvaffaqiyatsiz tool
   chaqiruvlari birinchi navbatda ko'rsatiladi. Har biri uchun ✅/✏️/🗑.
3. `uv run python scripts/build_personal_dataset.py` -- tasdiqlangan
   yozuvlardan (label=correct/corrected) **sana bo'yicha** bo'lingan
   `data/datasets/personal/{train,dev,test}.jsonl` quradi (oxirgi hafta =
   test, ma'lumot oqishining oldini olish uchun -- §3.2).

Bu uchtasi istalgan vaqt qayta ishga tushirilishi mumkin (masalan har hafta
ko'proq ma'lumot to'plagandan keyin `label_tool.py` + `build_personal_dataset.py`ni
qayta ishga tushirib, `run_training.sh`ni qayta boshlash).

## 3. Ochiq datasetlar + o'qitish (WSL ichida)

```bash
wsl -d Ubuntu
cd /mnt/c/ultron/training/stt
source ~/stt_train/venv/bin/activate   # yoki run_training.sh o'zi faollashtiradi
bash run_training.sh configs/lora_v1.yaml
```

To'rt bosqichni ketma-ket bajaradi (har biri alohida ham ishga tushirilishi
mumkin, `~/stt_train/` ichida):

1. **`prepare_open_datasets.py`** -- Mozilla Common Voice (uz) + Google
   FLEURS (uz_uz)ni yuklab, 16 kHz mono FLAC'ga o'giradi,
   `ultron.text.normalize.normalize_transcript` orqali o'tkazadi (Windows
   asosiy loyihaning `src/`idan import qilinadi -- §3.3'ning "bir xil
   ko'rinishga keltiriladi" talabi shu orqali bajariladi, ikkinchi marta
   yozilmaydi). FLEURS'ning **test** bo'limi mashg'ulotga aralashtirilmaydi --
   to'g'ridan-to'g'ri Windows'ning `data_dir`siga (`datasets/fleurs_uz_test/`)
   yoziladi, §3.5'dagi umumiy sifat regressiyasini tekshirish uchun.
   **Common Voice gated**: birinchi marta ishlatishdan oldin
   https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
   sahifasida shartlarga rozilik bildirib, `huggingface-cli login` qiling
   (yoki `HF_TOKEN` o'zgaruvchisini o'rnating). Bu qadam istamasangiz,
   `--common-voice-id` bilan gated bo'lmagan ko'zguga (masalan
   `fsicoli/common_voice_17_0`) ishora qiling -- litsenziyasini o'zingiz
   tekshiring.
2. **`build_mix_manifest.py`** -- shaxsiy (Windows'dagi
   `data/datasets/personal/`) + ochiq datasetlarni §3.3'dagi nisbatda
   (standart 50/50, `configs/lora_v1.yaml`'ning `mix.personal_ratio`si)
   aralashtiradi -- shaxsiy tomon kam bo'lgani uchun **oversample** qilinadi
   (ko'p marta takrorlanadi), ochiq tomon to'liq saqlanadi.
3. **`train_lora.py`** -- §3.4'dagi LoRA konfiguratsiyasi (`r=32, alpha=64`,
   `q/k/v/out_proj`), fp16 + gradient checkpointing + 8-bit AdamW bilan
   o'qitadi, har `eval_steps`da dev WER'ni o'lchab eng yaxshi checkpoint'ni
   saqlaydi. Natija: `~/stt_train/checkpoints/ultron-uz-v1/adapter/` (faqat
   LoRA og'irliklari, asosiy model emas).
4. **`merge_lora.py`** -- adapterni asosiy modelga birlashtiradi
   (`merge_and_unload()`) va to'g'ridan-to'g'ri Windows'ning `data_dir`siga
   (`/mnt/d/ultron_data/models/stt/ultron-uz-v1-merged/` yoki qayerga
   sozlangan bo'lsa) yozadi -- alohida ko'chirish kerak emas.

**8 GB VRAM sig'masa** (`configs/lora_v1.yaml` sharhida yozilgan): avval
`per_device_train_batch_size`ni kamaytiring/`gradient_accumulation_steps`ni
oshiring, keyin `training.load_in_8bit: true` (QLoRA uslubi, §3.4'ning o'z
ikkinchi fallback'i) yoki asosga `medium`/`small`ga o'tishni sinang.

## 4. Windows'da eksport va baholash

```powershell
cd C:\ultron
uv run python scripts/export_stt_ct2.py --merged-model D:/ultron_data/models/stt/ultron-uz-v1-merged --version 1
uv run python scripts/finetune_eval.py --new-model data/models/stt/ultron-uz-v1-ct2 --training-meta <adapter_dir>/training_meta.json
```

`export_stt_ct2.py` `scripts/download_models.py`ning o'zi ishlatgan
`ctranslate2.converters.TransformersConverter`ni chaqiradi (mahalliy papkaga
ishora qilib, HF'dan qayta yuklamasdan). `finetune_eval.py` uchta
to'plamda (TZ-03'ning 80 gaplik umumiy to'plami, shaxsiy test, FLEURS test)
eski va yangi modelni solishtiradi (`scripts/benchmark_stt.py`'ning
`run_one_config`'ini qayta ishlatib) va §3.5'ning qabul qoidasini o'zi
tekshiradi: (1) va (2)da yaxshilanish **va** (3)da 3 punktdan ko'p
yomonlashmaslik. O'tsa, `config.yaml`'ning `stt.model`i avtomatik yangilanadi
va `MODEL_CARD.md` yoziladi.

## 5. Bilinigan xavf nuqtalari (haqiqiy GPU'siz sinalmagan)

`docs/tz/10_wake_word_training.md`ning WSL skriptlari kabi, bu yerdagi
`train_lora.py`/`merge_lora.py`/`prepare_open_datasets.py` **haqiqiy
GPU/WSL/soatlab vaqt talab qiladigan muhitda sinalmagan** -- bu sessiyada na
GPU, na WSL, na bu kutubxonalarning o'zi mavjud edi (faqat sintaksis/import
tekshiruvi va soф-mantiq qismlari, masalan `build_mix_manifest.py`'ning
oversampling hisobi, qo'lda sinovdan o'tkazildi). Eng ehtimoliy birinchi
xato manbai kod mantig'i emas, balki **kutubxona versiyasi** (Whisper+PEFT
integratsiyasi `transformers`/`peft`ning yangi versiyalarida tez-tez
o'zgaradi):

| Xavf nuqtasi | Nima kutilmoqda | Agar farq qilsa |
|---|---|---|
| `Seq2SeqTrainer` + PEFT modeli | Zamonaviy `transformers`/`peft` PEFT-wrapped modelni to'g'ridan-to'g'ri qo'llab-quvvatlaydi (`label_names=["labels"]` bilan) | Eski uslubdagi `SavePeftModelCallback` kerak bo'lishi mumkin -- rasmiy PEFT-Whisper qo'llanmasiga qarang |
| `optim="adamw_bnb_8bit"` | `bitsandbytes` WSL2/CUDA'da muammosiz o'rnatiladi | Ba'zi CUDA versiyalarida `bitsandbytes` alohida `--extra-index-url` yoki manba kodidan qurishni talab qilishi mumkin |
| `load_in_8bit=True` (fallback) | `WhisperForConditionalGeneration.from_pretrained(..., load_in_8bit=True)` ishlaydi | Yangi `transformers`da `BitsAndBytesConfig(load_in_8bit=True)` orqali berish talab qilinishi mumkin |
| Common Voice gated kirish | Bir martalik `huggingface-cli login` yetarli | HF interfeysi o'zgargan bo'lsa, dataset sahifasidagi "Access" bo'limiga qarang |

## 6. Litsenziyalar (TZ'ning "litsenziya tekshiriladi" talabi)

- Mozilla Common Voice: **CC0** (public domain) -- cheklovsiz.
- Google FLEURS: **CC-BY-4.0** -- atribut talab qiladi (shaxsiy/tadqiqot
  foydalanish uchun muammo emas).
- Ikkalasi ham shaxsiy, tijoriy bo'lmagan fine-tune uchun mos.
