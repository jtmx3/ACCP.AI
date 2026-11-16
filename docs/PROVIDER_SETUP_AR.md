# إعداد ACCP.AI المحلي

## الافتراضي والحدود

يبدأ التطبيق بوضع `MODEL_PROVIDER=rules`: معالجة محلية بالقواعد المحدودة من دون استدعاء نموذج. لا يتصل التطبيق إلا بـOllama على loopback (`127.0.0.1` أو `localhost` أو `::1`) عند اختيار `MODEL_PROVIDER=ollama`. أي مزود بعيد أو عنوان غير محلي مرفوض. لا توجد مفاتيح API خارجية مطلوبة أو مدعومة.

النموذج المستهدف في إعداد المثال هو `Jais-3-3.7B-GGUF:Q4_K_M`. إعداد Ollama المطلوب `OLLAMA_REASONING_EFFORT=none` والمهلة `MODEL_TIMEOUT_SECONDS=120`. تهيئة الإعداد لا تثبت أن الخدمة أو النموذج موجودان، كما أن نجاح التوليد الثابت لا يثبت دقة الاستخراج أو صلاحية التقييم.

## Windows PowerShell

1. ثبّت Ollama وشغّله محليًا. من PowerShell تحقّق من النسخة، واسحب النموذج المستهدف إذا كان متاحًا ضمن سجل Ollama لديك، ثم أنشئ الاسم المحلي المخصص باستخدام Modelfile المشروع:

```powershell
ollama --version
ollama pull Jais-3-3.7B-GGUF:Q4_K_M
ollama create accp-jais -f Modelfile.jais
ollama list
```

يتطلب `FROM` في `Modelfile.jais` أن يكون الوسم الأساسي متاحًا محليًا. لا تعتبر `ollama create` ناجحًا ما لم ينجح الأمر ويظهر النموذج في `ollama list`. إن كان اسم الوسم مختلفًا في تثبيتك، اضبط `MODEL_NAME` على الاسم المحلي الظاهر حرفيًا، ولا تدّعِ نجاح الاتصال قبل فحص الحالة.

2. من PowerShell نفسه الذي سيشغّل التطبيق، فعّل البيئة واضبط المتغيرات. افتراضيًا اترك النموذج معطلًا بالقواعد؛ لتجربة Jais غيّر المزود صراحةً إلى Ollama:

```powershell
Set-Location C:\path\to\ACCP.AI
.\.venv\Scripts\Activate.ps1
$env:MODEL_PROVIDER = "ollama"
$env:MODEL_BASE_URL = "http://127.0.0.1:11434/v1"
$env:MODEL_NAME = "Jais-3-3.7B-GGUF:Q4_K_M"
$env:MODEL_TIMEOUT_SECONDS = "120"
$env:OLLAMA_REASONING_EFFORT = "none"
python -m app.rag --reindex
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

إذا أنشأت الاسم المحلي `accp-jais` بدل تشغيل الوسم الأساسي، استخدم `$env:MODEL_NAME = "accp-jais"` بعد التحقق من وجوده في `ollama list`.

3. افتح `http://127.0.0.1:8000`. يعرض فحص المزود الإعداد، وإمكان الوصول إلى قائمة النماذج، ومطابقة الوسم؛ لا يرسل نص الحالة. اختبار التوليد يرسل JSON ثابتًا غير متعلق بأي جلسة. لا تبدأ تجربة حوار قبل نجاح الفحص.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8000/api/provider/status | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/provider/test `
  -ContentType "application/json" -Body "{}" | ConvertTo-Json -Depth 5
Test-NetConnection 127.0.0.1 -Port 11434
```

## موافقات الإرسال

- استخراج الوقائع أو صياغة جواب إرشادي بالنموذج يحتاج موافقة لكل رسالة؛ موافقة استخراج الرسالة لا تفعل إرسال السياق السابق، فله خيار منفصل.
- يظهر خيار **صياغة سؤال المتابعة المحلي** تلقائيًا عند إعداد Ollama، ويمكن إخفاؤه عبر `ACCP_ADAPTIVE_MODEL_QUESTIONS=0`. لا يولّد التطبيق سؤالًا تكيفيًا حتى تمنح موافقة منفصلة لهذه الجلسة من الواجهة. يستخدم النموذج اسم الخانة وسبب السؤال والجزء غير المحسوم وخصائص تصنيفية محدودة، ويصوغ سؤالًا أردنيًا طبيعيًا بدل نسخ القالب حرفيًا؛ لا تُرسل نصوص المستخدم أو الاقتباسات أو معرفات الجلسات أو رموزها أو أرقام التواصل. المحرك يحدد الخانة التالية، والسؤال المتولد يخضع للمدقق؛ إذا فشل التوليد أو التحقق تعود صياغة القواعد.
- لا يُرسل سجل المحادثة كاملًا. لا ترسل كلمات مرور أو رموز تحقق أو صورًا أو أدلة. لا يرفع التطبيق بلاغًا ولا يتصل بجهة نيابة عن المستخدم.

## فحص استخراج تشخيصي حقيقي على Windows

بعد تشغيل Ollama والتأكد من النموذج، يمكن تشغيل الأمر التالي من جذر المستودع. يستخدم `generate_interview_extraction` ومدقق الوقائع الحقيقيين. الرسالة اصطناعية؛ الأمر لا يثبت جودة النموذج خارج هذه المحاولة:

```powershell
$env:MODEL_PROVIDER='ollama'; $env:MODEL_BASE_URL='http://127.0.0.1:11434/v1'; $env:MODEL_NAME='Jais-3-3.7B-GGUF:Q4_K_M'; $env:MODEL_TIMEOUT_SECONDS='120'; $env:OLLAMA_REASONING_EFFORT='none'; .\.venv\Scripts\python.exe scripts/diagnose_ollama_extraction.py --pending-slot narrative --message 'في شخص عم يهددني على واتساب.'
```

يعرض التشخيص الزمن، ومخرجات المدقق، والحقائق المقبولة وأسباب التصفية، دون إظهار الرد الخام. `valid_empty` تعني أن الرد صالح بنيويًا لكن لا يحوي وقائع؛ `all_facts_filtered` تعني أن المدقق استبعد المقترحات؛ و`extracted` تعني أن حقيقة واحدة على الأقل اجتازت المدقق. لا تحتسب الحالتين الأوليين نجاحًا للاستخراج، ولا تحتسب رجوع التطبيق إلى القواعد نجاحًا للنموذج.

## الاختبارات والتقييمات الاصطناعية

شغّل من جذر المشروع:

```powershell
python -m compileall app scripts tests
node --check static/app.js
python -m app.rag --reindex
python -m pytest -q --ignore=tests/test_browser.py
python scripts/evaluate_scenarios.py --mode rules --dataset tests/evaluation_scenarios.json --output $env:TEMP\accp-eval-rules.json
python -m scripts.evaluate_conversation_v2 --mode rules --suite both --output $env:TEMP\accp-conv-v2-rules.json
python -m scripts.evaluate_conversation_v3 --mode rules --suite all --output $env:TEMP\accp-conv-v3-rules.json
```

مجموعات الاختبار اصطناعية وليست مراجعة بشرية مستقلة. إن شُغّل وضع النموذج على جهاز يتوفر فيه Ollama، احفظ كل تقرير في ملف جديد ولا تستبدل التقارير التاريخية. سجّل نتائج النموذج الخام، وقرار المدقق، والدمج بالقواعد والرجوع كقياسات منفصلة. اختبارات الموفر الوهمية أو القواعد لا تثبت عمل Ollama حيًا.

## مراجع التشغيل

- [توثيق Ollama على Windows](https://docs.ollama.com/windows)
- [واجهة قائمة النماذج المحلية في Ollama](https://docs.ollama.com/api/tags)

لا يقدم هذا النموذج الأولي استشارة قانونية أو تحديدًا احتماليًا للخطر، ولا يثبت صدق الوقائع؛ راجع ملخص القيود داخل المشروع.
