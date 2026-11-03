# إعداد النموذج المحلي أو المزود المتوافق

**الافتراضي آمن ومحلي:** عند `MODEL_PROVIDER=rules` يعمل ACCP.AI بالقواعد المحلية، ولا يرسل رسائل إلى نموذج. إعداد المزود لا يثبت أن الخدمة متاحة أو أن النموذج موجود أو أن التوليد ينجح. تعرض الواجهة هذه المراحل منفصلة، ويمكن فحص القائمة دون إرسال بيانات الحالة، ثم إجراء اختبار توليد بنص ثابت.

يدعم Ollama على Windows تطبيقًا أصليًا يعمل كخدمة خلفية محلية. توثيق Ollama يذكر عنوان API المحلي `http://localhost:11434` وواجهة متوافقة جزئيًا مع OpenAI على `http://localhost:11434/v1`، ومنها `/v1/chat/completions` و`/v1/models` ووضع JSON. كما يتيح `GET /api/tags` تعداد النماذج المحلية. [1] [2] [3]

## Windows PowerShell — Ollama محلي

1. ثبّت Ollama من دليل Windows الرسمي، ثم شغّله واتركه يعمل في الخلفية. يتطلب Windows 10 22H2 أو أحدث بحسب دليل المزود. اسحب نموذجًا مناسبًا لسعة الجهاز؛ `qwen3:4b` مثال على وسم منشور في مكتبة Ollama وليس توصية دقة أو ملاءمة للفرز الأمني. [1] [4]

```powershell
ollama --version
ollama pull qwen3:4b
ollama list
```

2. في PowerShell نفسه الذي ستشغّل منه التطبيق، اضبط المزود المحلي واسم النموذج كما يظهر حرفيًا في `ollama list`:

```powershell
Set-Location C:\path\to\accp-ai
.\.venv\Scripts\Activate.ps1
$env:MODEL_PROVIDER = "ollama"
$env:MODEL_BASE_URL = "http://127.0.0.1:11434/v1"
$env:MODEL_NAME = "qwen3:4b"
$env:MODEL_TIMEOUT_SECONDS = "120"
python -m app.rag --reindex
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

لا يلزم `MODEL_API_KEY` لموائم Ollama المحلي؛ يرسل الموائم قيمة محلية شكلية لأن العميل المتوافق يتوقع ترويسة، ويتجاهلها خادم Ollama المحلي. لا تستخدم هذا الافتراض مع خدمة Ollama سحابية أو مزود بعيد. [2]

3. افتح `http://127.0.0.1:8000`، ثم استخدم لوحة **مسار التشغيل الفعلي**. زر الفحص يختبر الإعداد واتصال قائمة النماذج وتطابق الاسم؛ لا يرسل رسالة حالة ولا يشغّل توليدًا. بعد نجاحه، زر اختبار التوليد يرسل نص JSON ثابتًا لا يتضمن بيانات المستخدم.

يمكن أيضًا فحص المراحل من PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8000/api/provider/status | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/provider/test `
  -ContentType "application/json" -Body "{}" | ConvertTo-Json -Depth 5
Test-NetConnection 127.0.0.1 -Port 11434
```

`/api/health` يعرض الإعداد المختار فقط. `/api/provider/status` يجري فحصًا غير توليدي للخدمة وقائمة النماذج. `/api/provider/test` لا يجري التوليد إلا بعد توافر الإعداد والخدمة والنموذج. النتيجة الناجحة تعني أن اختبار JSON الثابت نجح، ولا تثبت جودة الاستخراج العربي أو السلامة أو صلاحية المنتج.

## مزود OpenAI-compatible خارجي

اضبط قيم المزود في عملية التطبيق نفسها، ولا تضع مفتاحًا حقيقيًا في Git أو ملف مشترك:

```powershell
$env:MODEL_PROVIDER = "openai-compatible"
$env:MODEL_BASE_URL = "https://your-provider.example/v1"
$env:MODEL_NAME = "<model-id-supported-by-your-endpoint>"
$env:MODEL_API_KEY = "<secret-from-your-secret-store>"
$env:MODEL_TIMEOUT_SECONDS = "60"
```

أعد تشغيل التطبيق بعد تغيير المتغيرات. لا تُعرض حالة «الخدمة متاحة» حتى ينجح فحص القائمة، ولا تُعرض «توليد ناجح» حتى ينجح الاختبار الثابت. فشل الشبكة أو المهلة أو رفض المخطط يظهر في الواجهة، ويعود استخراج الوقائع إلى القواعد المحلية أو إجابة الإرشاد المستندة إلى المصادر المحلية.

## تقييم القواعد أو النموذج على الحالات الاصطناعية

شغّل الأوامر من جذر المستودع وفي نافذة PowerShell نفسها التي تحتوي إعداد المزود. تقييم القواعد محلي ولا يستدعي نموذجًا؛ أما `--mode model` فينفّذ فحص المزود وتوليدًا ثابتًا ثم يرسل الرسائل الاصطناعية الثابتة المحددة في مجموعة الاختبار، ولا يستخدم حالات أشخاص حقيقيين. قد يستغرق ذلك عدة دقائق أو يستهلك حصة المزود الخارجي. استخدم ملف إخراج مستقلًا لكل وضع، ولا تمرر مفتاحًا ضمن الأمر أو تحفظه في Git.

```powershell
python scripts/evaluate_scenarios.py --mode rules `
  --dataset tests/evaluation_scenarios.json `
  --output reports/evaluation/rules-windows.json

python scripts/evaluate_scenarios.py --mode model `
  --dataset tests/evaluation_scenarios.json `
  --output reports/evaluation/model-windows.json
```

يفشل وضع النموذج بوضوح إذا لم يكتمل الإعداد أو لم تصل الخدمة أو لم يظهر النموذج أو فشل التوليد. يسجل JSON المزود واسم النموذج والاستدعاءات المقبولة والمرفوضة والرجوع وزمن الاستجابة. لا يحتسب النظام نجاح الرجوع المحلي نجاحًا للنموذج؛ معيار اجتياز النموذج أشد من نجاح النظام المتكامل. استخدم فقط مجموعة الاصطناعية الثابتة إلى أن تتوفر موافقة وحوكمة واختبارات بيانات مناسبة.

## حدود الإرسال والموافقة

فحص الحالة غير التوليدي لا يرسل محتوى الحالة. واختبار التوليد يستخدم سؤالًا ثابتًا. أما استخراج الوقائع أو اختيار اقتباس لمعلومة إرشادية فيحتاج موافقة منفصلة لكل طلب. يرسل مسار الاستخراج الرسالة الحالية فقط، مع خانة السؤال المعلّقة؛ ولا يرسل سجل الجلسة أو الرسائل السابقة أو معرّف الجلسة أو الرمز. يرسل مسار السؤال الحالي ومقتطفات المصادر المختارة عند موافقة المستخدم عليه. إذا كتب المستخدم سؤالًا واضحًا في وضع المقابلة، يفهمه التطبيق ويجيب محليًا دون اعتبار موافقة استخراج الوقائع موافقة على إرسال سؤال إلى مزود خارجي.

المخرجات لا تحل محل القواعد: الاقتباسات يجب أن تطابق نص المصدر حرفيًا، وحقائق النموذج يجب أن تحمل اقتباسًا حرفيًا من الرسالة الحالية وأن تجتاز المخطط والتحقق المحلي. لا يُحوّل نجاح الاختبار الثابت إلى ادعاء بصحة فرز أو بنصيحة قانونية.

## المراجع

[1]: https://docs.ollama.com/windows "Ollama on Windows"
[2]: https://docs.ollama.com/api/openai-compatibility "Ollama OpenAI compatibility"
[3]: https://docs.ollama.com/api/tags "Ollama list models API"
[4]: https://ollama.com/library/qwen3/tags "Qwen3 model tags in the Ollama library"
