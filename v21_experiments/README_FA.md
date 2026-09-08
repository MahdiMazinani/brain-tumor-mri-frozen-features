# کدهای تکمیلی هایلایت‌های مقاله v21

## محدوده و وضعیت

این بسته فقط آزمایش‌ها و تحلیل‌های اضافه را انجام می‌دهد. مقاله، داده‌ها، native_cache، manifestها و نتایج کامل‌شده قبلی را تغییر نمی‌دهد. نه عدم‌توازن مصنوعی ساخته می‌شود، نه oversampling/undersampling و نه deduplication اجرا می‌شود. بررسی تکرار شناسه/مسیر برای صحت اتصال رکوردها، جست‌وجوی تصاویر تکراری نیست.

**این آزمایش‌ها بعد از مشاهده نتایج قبلی پیشنهاد شده‌اند؛ بنابراین همگی تکمیلی و exploratory هستند.** همان تست‌های قدیمی با یک فایل freeze جدید به تست تاریخیِ دست‌نخورده تبدیل نمی‌شوند. تقسیم بیمار نیز دقیقاً بر اساس شناسه‌های منتشرشده است، نه تضمین هویت افراد. هیچ شناسه‌ای merge نمی‌شود.

### تطبیق با هایلایت‌ها

- 3.8: class_weight کنترل قبلی از frozen_selection.json خوانده می‌شود؛ در پنج فایل ارسال‌شده balanced بود. کنترل‌های جدید: LP بدون PCA و LP با PCA95، هر دو با C در {0.01, 0.1, 1, 10} و همان LogisticRegression. هرکدام در هر فولد مستقل روی valid انتخاب می‌شوند. این مقایسه اثر افزودن PCA در یک پروتکل تنظیم‌شده مشترک است؛ لزوماً C منتخب یکسان نمی‌شود.
- 3.9 / 4.2: McNemar دقیق، شمارش زوج‌های ناسازگار، bootstrap خوشه‌ای شناسه‌ای با 2000 بازنمونه، CI معیارهای تجمیعی و اختلاف زوجی. McNemar اسلایسی فقط تشخیصی است و برای نتیجه‌گیری معنی‌داری بیمار استفاده نمی‌شود.
- زمان: بازیابی فیلدهای واقعاً ثبت‌شده؛ زمان استخراج و کل جست‌وجوی تاریخی که ثبت نشده‌اند null باقی می‌مانند. زمان‌گیری جدید اختیاری و جداگانه موجود است.
- خط مبنای آموزش کامل: ResNet-18 از ImageNet، همه پارامترها قابل آموزش، همان manifestها و preprocessing. این baseline جدید است، نه بازتولید کامل یک مقاله دیگر.
- سطح بیمار: majority vote برای هر هشت روش از CSV موجود، بدون آموزش. mean probability فقط وقتی فایل واقعاً ستون‌های p_0 تا p_(K-1) معتبر دارد؛ در CSV و final_results ارسالی احتمال‌ها وجود نداشتند.
- affiliation، نویسنده مسئول، funding، URL مخزن و DOI و صحت کتاب‌نامه آزمایش نیستند و این کد آن‌ها را تولید یا منتشر نمی‌کند.

## نصب بدون تغییر محیط قدیمی

ZIP را در پوشه پروژه استخراج کن تا مسیر `v21_experiments\NEW_CONTROLS.py` ایجاد شود. پوشه‌های `cv_figshare_02` و `runs_adaptive` را جابه‌جا نکن. از همان Python استفاده کن؛ upgrade کردن پکیج‌ها لازم نیست و می‌تواند مهرهای اجرای قبلی را نامعتبر کند.

دستورات زیر برای **CMD یا ترمینال PyCharm با cmd.exe** هستند، نه PowerShell. هیچ‌کدام را هم‌زمان با یک آموزش دیگر اجرا نکن.

```bat
cd /d C:\Users\mzx13\PycharmProjects\article2
venv311\Scripts\python.exe -c "import torch,torchvision,sklearn,numpy,pandas,joblib; print('torch',torch.__version__,'vision',torchvision.__version__,'sklearn',sklearn.__version__,'CUDA',torch.cuda.is_available())"
venv311\Scripts\python.exe -m unittest discover -s v21_experiments -p test_v21.py -v
```

تست‌ها از fixtureهای موقت ساختگی برای بررسی نرم‌افزار استفاده می‌کنند؛ این‌ها نتایج آزمایش MRI و سناریوی عدم‌توازن نیستند. این محیط sandbox فاقد scikit-learn و PyTorch بود: 14 تست غیرآموزشی پاس شدند، تست کامل LP skip شد، فایل‌های آموزش syntax-check شدند اما GPU/fit واقعی اینجا اجرا نشد. تست کامل LP را دستور بالا در venv311 اجرا می‌کند؛ در آن محیط انتظار می‌رود هیچ تستی skip نشود. آموزش ResNet باید روی سیستم خودت ارزیابی شود؛ تضمین نبود خطای محیطی نداریم.

## 1. تحلیل OOF بدون آموزش

```bat
venv311\Scripts\python.exe v21_experiments\OOF_ANALYSIS.py --oof "cv_figshare_02\cv_oof_predictions.csv" --out "v21_stats_01" --resamples 2000
```

ورودی واقعی بررسی‌شده: 3064 اسلایس، 233 شناسه، 8 روش. خروجی‌ها:

- metric_intervals.csv: accuracy، balanced accuracy، macro-F1 و CI نقطه‌ای 95% برای pooled slices و majority vote سطح شناسه، برای هر روش.
- paired_differences.csv: اختلاف proposed minus linear_probe و CI آن با بازنمونه‌های مشترک.
- mcnemar.csv: شمارش‌های discordant و p دوطرفه دقیق. ردیف slice_IID_DIAGNOSTIC_ONLY معنی‌داری معتبر در سطح بیمار نیست.
- patient_predictions.csv: 233 تصمیم برای هر روش، تعداد اسلایس، fold و پرچم tie. در تساوی رأی، کمترین کد کلاس انتخاب می‌شود؛ این قانون ثابت و تعداد tie باید گزارش شود.
- fold_metrics.csv و fold_mean_sd.csv: معیارهای هر فولد و mean±sample SD.
- analysis.json: hash ورودی، پروتکل، محدودیت‌ها و وجود/نبود احتمال‌ها.

اعداد فایل‌ها **کسر 0 تا 1** هستند؛ برای درصد ×100. CI pooled accuracy را کنار mean-fold accuracy با عنوان CI همان میانگین قرار نده. Bootstrap از شناسه‌ها با حفظ تعداد شناسه هر کلاس (class-stratified) و جایگذاری کامل اسلایس‌های شناسه انتخاب‌شده استفاده می‌کند. intervalها مشروط بر همین پیش‌بینی‌های ذخیره‌شده‌اند و عدم‌قطعیت آموزش مجدد/انتخاب مدل و وابستگی مدل‌های فولدهای همپوشان را کامل پوشش نمی‌دهند.

نتیجه اجرای همین اسکریپت روی CSV قبلی در `existing_oof_analysis` داخل ZIP است. برای ورودی به‌روز خودت دستور بالا را اجرا کن. رأی اکثریت با میانگین احتمال یکی نیست؛ برای ساخت دومی از برچسب سخت احتمال جعل نکن.

## 2. کنترل‌های تازه: ابتدا توسعه همه روش‌ها، بعد تست

**ترتیب توصیه‌شده:** init هر دو خانواده، develop هر دو خانواده تا پایان همه فولدها، سپس evaluate هر دو. قبل از کامل‌شدن توسعه همه خانواده‌های برنامه‌ریزی‌شده، نتایج تست جدید را نبین و تنظیمات را بر اساس آن تغییر نده.

### 2A. دو کنترل Linear Probe روی همان 5 فولد

```bat
venv311\Scripts\python.exe v21_experiments\NEW_CONTROLS.py init --kind lp --plan "cv_figshare_02" --out "v21_lp_cv_01"
venv311\Scripts\python.exe v21_experiments\NEW_CONTROLS.py develop --out "v21_lp_cv_01"
```

هر فولد 2 حالت PCA × 4 مقدار C = 8 fit جدید دارد؛ مجموعاً 40 fit LogisticRegression، نه اجرای مجدد 210 کاندید قدیمی. داده‌های ویژگی دقیقاً از native_cache همان فولد خوانده می‌شوند و token مبتنی بر manifest و preprocessing و labels بررسی می‌شود. اگر cache دقیق موجود نباشد برنامه **متوقف می‌شود**؛ cache مشابه یا قدیمی دیگری را حدس نمی‌زند. در این وضعیت متن خطا و فهرست فایل‌های native_cache همان فولد را بفرست؛ نتایج اصلی را دوباره اجرا یا حذف نکن.

StandardScaler فقط روی train، PCA95 فقط روی train با full SVD، LogisticRegression با solver=lbfgs، max_iter=3000، class_weight=balanced و seed42. انتخاب با valid balanced accuracy؛ تصمیم argmax؛ no train+valid refit. دو مدل منتخب هر فولد ذخیره می‌شوند؛ test در مرحله develop بارگذاری نمی‌شود. کنترل convergence فعال است.

### 2B. ResNet-18 fine-tuned روی همان 5 فولد

وزن ImageNet احتمالاً از قبل روی سیستم هست. ابتدا بررسی کن؛ فقط اگر missing بود دستور download را اجرا کن (فقط دانلود وزن رسمی torchvision است).

```bat
venv311\Scripts\python.exe v21_experiments\BENCHMARK_TIMING.py weights --backbones resnet18
venv311\Scripts\python.exe v21_experiments\BENCHMARK_TIMING.py weights --backbones resnet18 --download
venv311\Scripts\python.exe v21_experiments\NEW_CONTROLS.py init --kind e2e --plan "cv_figshare_02" --out "v21_r18_cv_01" --epochs 20 --patience 5 --batch 16 --device cuda
venv311\Scripts\python.exe v21_experiments\NEW_CONTROLS.py develop --out "v21_r18_cv_01"
```

پروتکل ثابت: ResNet18 ImageNet1K_V1؛ تمام لایه‌ها از ابتدا قابل آموزش؛ لایه خروجی 3کلاسه؛ AdamW، weight_decay=1e-4، learning rates={1e-4,1e-3}؛ حداکثر 20 epoch برای هر نرخ؛ توقف بعد از 5 epoch بدون بهبود valid BA؛ best epoch و LR فقط روی valid؛ در تساوی اولین گزینه. class-weighted cross entropy از فراوانی train. RGB، bilinear resize224 با antialias، ImageNet normalization، بدون augmentation، AMP، TTA یا sampling مصنوعی. Batch16، num_workers0 برای Windows، seed42، deterministic algorithms. هیچ dropout یا loss با توجه به تست تنظیم نمی‌شود.

حداکثر 5×2×20=200 epoch، اغلب کمتر با early stopping. زمان اجرا به سیستم وابسته است. به اندازه LP سریع نیست و **باید تصاویر train را واقعاً دوباره آموزش بدهد**؛ از frozen features نمی‌توان fine-tuning انتهابه‌انتها کرد. دانلود وزن در تایمر توسعه حساب نمی‌شود؛ GPU/نسخه‌ها و زمان هر epoch ذخیره می‌شوند. نام این کنترل را «ResNet-18 fine-tuned under the matched fold protocol» بگذار، نه «exact reproduction». بودجه جست‌وجوی آن با روش پیشنهادی یکسان نیست.

اگر OOM رخ داد: پیش از شروع هر test، یک خروجی جدید برای کل خانواده با batch8 بساز و تمام فولدهای آن خانواده را با همان تنظیم اجرا کن. مخلوط batch/configهای مختلف را بی‌توضیح جمع نکن.

### 2C. تست نهایی کنترل‌های جدید، بعد از اتمام هر دو develop

```bat
venv311\Scripts\python.exe v21_experiments\NEW_CONTROLS.py evaluate --out "v21_lp_cv_01"
venv311\Scripts\python.exe v21_experiments\NEW_CONTROLS.py evaluate --out "v21_r18_cv_01"
```

هر خانواده پیش از test همه فولدهای خودش را seal می‌کند. برای هر فولد فقط دو LP منتخب یا یک ResNet منتخب evaluate می‌شوند. هیچ مدل بر اساس نمره outer test انتخاب نمی‌شود. فایل‌های کلیدی:

- new_controls_fold_metrics.csv
- new_controls_summary.csv (mean و sample SD؛ برای تک‌فولد SD خالی است)
- new_controls_oof.csv (احتمال‌های واقعی p_0... و برچسب/شناسه)
- new_controls_timing.csv (زمان اجرای جدید، نه زمان تاریخی)
- CONFIG.json، DEVELOPMENT_SEAL.json و fold_*/DEVELOPED.json
- fold_*/attempt_*/validation_trials.csv برای LP و epoch_log.csv برای ResNet.

### ادامه اجرا بدون از بین بردن خروجی‌های کامل‌شده

- اجرای دوباره `develop` فولدهای کامل همین کنترل جدید را KEEP می‌کند و فقط فولدِ تکمیل‌نشده را از ابتدا در attempt جدید اجرا می‌کند؛ resume بین epochها نیست. attempt قبلی باقی می‌ماند.
- اجرای دوباره `evaluate` خروجی‌های کامل و hash-verified را KEEP می‌کند. اگر FINAL_STARTED وجود داشته ولی FINAL_DONE نداشته باشد، متوقف می‌شود تا خطای تست بررسی شود؛ نشانگر را پاک نکن.
- ACTIVE.lock مانع اجرای هم‌زمان است. بعد از قطع غیرعادی فقط با اطمینان از نبود هیچ پردازش فعال می‌توان همین lock را دستی پاک کرد؛ هیچ seal یا FINAL_STARTED را پاک نکن.
- وجود نتایج قبلی fold1 باعث skip مدل تازه نمی‌شود: مدل تازه باید روی هر پنج فولد آموزش ببیند. تنها مراحل قدیمی اصلاً دست زده نمی‌شوند.

## 3. اتصال نتایج برای جدول تکمیلی

```bat
venv311\Scripts\python.exe v21_experiments\MERGE_OOF.py --inputs "cv_figshare_02\cv_oof_predictions.csv" "v21_lp_cv_01\new_controls_oof.csv" "v21_r18_cv_01\new_controls_oof.csv" --out "v21_combined\oof.csv"
venv311\Scripts\python.exe v21_experiments\OOF_ANALYSIS.py --oof "v21_combined\oof.csv" --out "v21_combined\stats_proposed_vs_linear" --resamples 2000
```

این بار معیارهای همه 11 روش گزارش می‌شود، اما آزمون زوجی همچنان فقط زوج اصلی proposed و linear_probe است. برای خانواده مقایسه‌های اضافی، همه زوج‌های موردنظر و اصلاح چندگانگی باید مشخص شوند؛ زوج برنده را بعد از دیدن p انتخاب نکن. از آنجا که فایل قدیمی probability ندارد، فایل ترکیبی probability را حذف می‌کند و فقط pooling اکثریت دارد؛ فایل‌های اصلی probability جدید حفظ می‌شوند. برای مقایسه دو LP جدید و هر دو pooling می‌توان جدا اجرا کرد:

```bat
venv311\Scripts\python.exe v21_experiments\OOF_ANALYSIS.py --oof "v21_lp_cv_01\new_controls_oof.csv" --out "v21_lp_cv_01\posthoc_pooling" --a lp_tuned_raw --b lp_tuned_pca95
```

هر تحلیل اضافی اکتشافی است؛ نباید با چند اجرای مختلف دنبال p معنادار رفت.

## 4. زمان اجرا

### بازیابی اطلاعات موجود، بدون آموزش

```bat
venv311\Scripts\python.exe v21_experiments\BENCHMARK_TIMING.py audit --root "cv_figshare_02" --out "v21_old_timing_audit"
```

جمع fit_seconds همه ردیف‌های قدیمی معتبر نیست: زمان treeهای reuseشده و ensembleها می‌تواند چندبار در ردیف‌ها تکرار شده باشد. اسکریپت فیلدها را با مسیر منبع استخراج می‌کند ولی total wall-clock و mean unique-fit تاریخی را جعل نمی‌کند. تغییر زمان فایل‌ها هم شاهد زمان پردازش نیست.

### اندازه‌گیری تازه استخراج روی کل یک دیتاست (اختیاری؛ بدون آموزش)

```bat
venv311\Scripts\python.exe v21_experiments\BENCHMARK_TIMING.py weights --download
venv311\Scripts\python.exe v21_experiments\BENCHMARK_TIMING.py extraction --manifest "cv_figshare_02\manifests\fold_01" --out "v21_extract_figshare" --device cuda --batch 16 --repeats 3
venv311\Scripts\python.exe v21_experiments\BENCHMARK_TIMING.py extraction --manifest "prepared\dataset_02" --out "v21_extract_7023" --device cuda --batch 16 --repeats 3
```

train+valid+test یک فولد کل دیتاست را پوشش می‌دهد؛ برای اندازه‌گیری استخراجِ یک بار کل دیتاست لازم نیست همین 3064 تصویر در پنج manifest تکرار شود. هر تکرار ویژگی‌ها را واقعاً محاسبه و دور می‌ریزد؛ native_cache را نمی‌خواند/نمی‌نویسد. تایمر شامل خواندن تصاویر، transform، forward، انتقال به CPU؛ GPU sync دارد. مدل‌سازی و نوشتن فایل feature از تایمر اصلی جدا/حذف شده‌اند. OS cache ریست نمی‌شود، پس «cold-disk» ننام. این تنظیمات و scope را همراه نتایج گزارش کن.

### زمان کل یک grid تازه در هر فولد (اختیاری و پرهزینه)

اگر ادعای زمان کل جست‌وجوی اصلی ضروری است و لاگ قبلی کافی نیست، باید measurement جدید انجام شود. کد زیر از ADAPTIVE_RUN موجود فقط مرحله develop را در پوشه خالی جدید اجرا می‌کند و elapsed time از شروع تا پایان child process را می‌سنجد؛ هیچ test score تولید نمی‌کند. این دستور پیش‌بینی‌های قبلی را پاک نمی‌کند، ولی عمداً محاسبات اصلی را برای benchmark زمان تکرار می‌کند. جزو مسیر لازم LP/ResNet نیست.

```bat
venv311\Scripts\python.exe v21_experiments\BENCHMARK_TIMING.py grid --project "." --manifest "cv_figshare_02\manifests\fold_01" --out "v21_grid_time_fold01"
```

برای تمام پنج فولد، فایل `TIME_ALL_FOLDS.bat` داخل همین بسته را از ریشه پروژه اجرا کن. این مسیر wrapper عادی ADAPTIVE_RUN است، نه لزوماً نسخه fast-resume قبلی؛ عدد آن باید با همین نام و دامنه معرفی شود. زمان‌های قبلی یا speedup نسبت به end-to-end را از آن استنتاج نکن. grid.log لاگ پردازش است؛ مجموع wall-clock شامل extraction، fitting، selection و IO است.

## 5. اگر LPهای جدید را روی 7023 هم لازم داشتی

```bat
venv311\Scripts\python.exe v21_experiments\NEW_CONTROLS.py init --kind lp --manifest "prepared\dataset_02" --run "runs_adaptive\experiment_02" --out "v21_lp_7023_01"
venv311\Scripts\python.exe v21_experiments\NEW_CONTROLS.py develop --out "v21_lp_7023_01"
venv311\Scripts\python.exe v21_experiments\NEW_CONTROLS.py evaluate --out "v21_lp_7023_01"
```

این تک holdout است، نه پنج‌فولد و نه patient-level. OOF_ANALYSIS را برای ادعای سطح بیمار روی این داده بدون PID استفاده نکن. اجرای تست را تنها بعد از freeze همه گزینه‌هایی که از قبل برای این داده برنامه‌ریزی کرده‌ای انجام بده.

## پیش از بازنویسی مقاله

1. پاراگراف 3.9 اکنون انجام‌شدن آزمون را پیشاپیش بیان می‌کند؛ فقط نتایج واقعی اسکریپت را جایگزین کن. برای ادعای superiority، McNemar اسلایسی را شاهد مستقل بیمار ندان.
2. نسخه PyTorch نوشته‌شده در v21 برابر 2.1 است؛ رکورد قبلی 2.5.1+cu121 بود. مشخصات فعلی از CONFIG.json و مشخصات اجرای قدیمی از freeze همان اجرا نقل شوند؛ سخت‌افزار تأییدنشده را فرض نکن.
3. همه زمان‌ها دامنه، کش، batch، دستگاه و نوع اجرای جدید/قدیمی مشخص داشته باشند؛ زمان fit کنترل LP جای زمان grid اصلی نیست.
4. جزئیات منابع هایلایت‌شده باید با مقاله/DOI اصلی راستی‌آزمایی شوند. چند مشخصه با نسخه ارسالی قبلی تفاوت دارند؛ این کد صحت آن‌ها را تأیید نمی‌کند.
5. نتایج تکمیلی نباید وارد انتخاب مجدد pipeline اصلی شوند. اگر نتیجه منفی بود همان را گزارش کن.

## فایل‌هایی که بعد از اجرا بفرستی

پوشه‌های v21_stats_01، فایل‌های new_controls_* و CONFIG.json، fold_*/DEVELOPED.json، validation_trials.csv و epoch_log.csv هر دو خانواده، و timing CSV/JSONهای اجراشده. checkpointهای .pt و .joblib و feature cacheها برای بررسی متن مقاله لازم نیست ارسال شوند.
