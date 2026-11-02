(() => {
  const $ = (id) => document.getElementById(id);
  const STORAGE_KEY = 'accp.local.cases.v1';
  const labels = {
    narrative: 'وصف الواقعة', threat: 'وجود تهديد', impersonation: 'انتحال شخصية', platform: 'المنصة', details: 'تفاصيل الواقعة',
    additional_details: 'تفاصيل إضافية', publication_threat: 'تهديد بالنشر', direct_danger: 'الخطر المباشر الحالي',
    active_unauthorized_funds: 'أموال معرضة لتصرف دون إذن', fraud_description: 'وصف خداع مالي',
    initial_time: 'وقت البداية', latest_time: 'أحدث تواصل',
    repeated: 'استمرار أو تكرار التواصل', requested_amount: 'المبلغ المطلوب', paid_amount: 'المبلغ المدفوع',
    payment_method: 'وسيلة الدفع', deadline: 'مهلة الدفع', published: 'حدوث النشر', identity: 'هوية الشخص',
    contact: 'معرّف التواصل', evidence: 'الأدلة المذكورة', account_access: 'دخول غير مصرح به'
  };
  let current = null;
  let session = null;
  let kind = 'answer';
  let busy = false;

  function savedCases() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); }
    catch { return []; }
  }
  function saveCases(items) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
    renderCaseList();
  }
  function renderCaseList() {
    const list = $('case-list');
    list.replaceChildren(new Option('اختر جلسة محفوظة', ''));
    for (const item of savedCases()) list.add(new Option(`${item.label || 'إفادة'} · ${item.id.slice(0, 8)}`, item.id));
  }
  function setBusy(value) {
    busy = value;
    $('send').disabled = value;
    $('start-case').disabled = value;
    $('new-case').disabled = value;
  }
  function status(text = '', ok = false) {
    const element = $('status-message');
    element.textContent = text;
    element.style.color = ok ? '#075a56' : '';
  }
  async function api(path, options = {}) {
    const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
    if (current?.token) headers['X-Case-Token'] = current.token;
    const response = await fetch(path, { ...options, headers });
    let payload;
    try { payload = await response.json(); }
    catch { payload = { detail: await response.text() }; }
    if (!response.ok) {
      const detail = payload.detail;
      const error = new Error(typeof detail === 'string' ? detail : (detail?.message || 'تعذر إكمال الطلب.'));
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    return payload;
  }
  function addBubble(message) {
    const bubble = document.createElement('article');
    bubble.className = `bubble ${message.role === 'user' ? 'user' : 'assistant'}`;
    const text = document.createElement('div');
    text.textContent = String(message.text ?? '');
    bubble.append(text);
    if (message.citations?.length) {
      const links = document.createElement('div');
      links.className = 'citation-list';
      for (const citation of message.citations) {
        const anchor = document.createElement('a');
        anchor.href = citation.url;
        anchor.target = '_blank';
        anchor.rel = 'noopener noreferrer';
        anchor.textContent = `${citation.publisher}: ${citation.title}`;
        links.append(anchor);
      }
      bubble.append(links);
    }
    const meta = document.createElement('div');
    meta.className = 'bubble-meta';
    meta.textContent = message.created_at ? new Date(message.created_at).toLocaleString('ar') : '';
    bubble.append(meta);
    $('chat').append(bubble);
  }
  function renderSummary(state) {
    const panel = $('review');
    const show = ['review', 'urgent_stop'].includes(state.stage);
    panel.classList.toggle('hidden', !show);
    if (!show) return;
    const assessment = state.assessment || {};
    $('priority-chip').textContent = `أولوية المراجعة: ${assessment.priority_score == null ? 'غير محددة' : `${assessment.priority_score}/100`} · ${assessment.priority_label || ''}`;
    $('assessment-reasons').textContent = `${assessment.reasons?.length ? `الأسباب: ${assessment.reasons.join('؛ ')}` : 'لم تتوفر إشارة كافية لتحديد الأولوية.'}\nاكتمال المعلومات: ${assessment.completeness_percent ?? 0}% — ${assessment.completeness_note || ''}\nجودة الاستخراج: ${assessment.extraction_quality || 'غير متاحة'}؛ لا يوجد تحقق مستقل.`;
    const factsList = $('facts-list');
    factsList.replaceChildren();
    const facts = state.facts || {};
    const order = ['narrative', 'threat', 'impersonation', 'platform', 'details', 'additional_details', 'publication_threat', 'direct_danger', 'active_unauthorized_funds', 'fraud_description', 'initial_time', 'latest_time', 'repeated', 'requested_amount', 'paid_amount', 'payment_method', 'deadline', 'published', 'identity', 'contact', 'evidence', 'account_access'];
    for (const key of order) {
      const fact = facts[key];
      if (!fact) continue;
      const wrapper = document.createElement('div');
      wrapper.className = 'fact-item';
      const term = document.createElement('dt');
      term.textContent = labels[key] || key;
      const description = document.createElement('dd');
      let value = fact.value;
      if (fact.status === 'denied') value = 'منفي صراحة بحسب الإفادة';
      else if (fact.status === 'unknown') value = 'المستخدم لا يعرف';
      else if (fact.status === 'not_remembered') value = 'المستخدم لا يتذكر';
      else if (fact.status === 'declined') value = 'فضّل عدم الإجابة';
      else if (fact.status === 'unresolved') value = 'لم تُحسم بعد محاولة التوضيح';
      else if (fact.status === 'conflict') value = 'تعارض يحتاج توضيحًا';
      else if (typeof value === 'boolean') value = value ? 'نعم بحسب الإفادة' : 'لا بحسب الإفادة';
      description.textContent = value == null ? 'غير محدد بحسب الإفادة' : String(value);
      wrapper.append(term, description);
      if (fact.quote) {
        const quote = document.createElement('div');
        quote.className = 'fact-quote';
        quote.textContent = `الاقتباس: «${fact.quote}»`;
        wrapper.append(quote);
      }
      for (const alternative of fact.alternatives || []) {
        const item = document.createElement('div');
        item.className = 'fact-quote';
        item.textContent = `إفادة بديلة: ${alternative.value ?? alternative.status} — «${alternative.quote || ''}»`;
        wrapper.append(item);
      }
      for (const previous of fact.history || []) {
        if (!previous.quote) continue;
        const item = document.createElement('div');
        item.className = 'fact-quote';
        item.textContent = `سجل سابق: ${previous.value ?? previous.status} — «${previous.quote}»`;
        wrapper.append(item);
      }
      for (const support of fact.supporting_statements || []) {
        const item = document.createElement('div');
        item.className = 'fact-quote';
        item.textContent = `إفادة داعمة: «${support.quote || ''}»`;
        wrapper.append(item);
      }
      factsList.append(wrapper);
    }
    $('confirmation-state').textContent = state.confirmed ? 'تم تأكيد إصدار الملخص المعروض. أي تعديل لاحق يتطلب مراجعة جديدة.' : 'لم يتم تأكيد الملخص بعد.';
    $('confirm-summary').disabled = !!state.confirmed;
  }
  function renderState(state) {
    session = state;
    $('welcome').classList.add('hidden');
    $('case-view').classList.remove('hidden');
    $('case-title').textContent = `الحالة ${state.session_id.slice(0, 8)}`;
    $('stage-badge').textContent = ({ interview: 'مقابلة', paused: 'محفوظة', review: 'مراجعة', urgent_stop: 'سلامة أولًا' })[state.stage] || 'جلسة';
    $('pause-case').textContent = state.stage === 'paused' ? 'استكمال المقابلة' : 'حفظ ومتابعة لاحقًا';
    $('chat').replaceChildren();
    for (const message of state.messages || []) addBubble(message);
    $('jurisdiction').value = state.jurisdiction || '';
    renderSummary(state);
    $('chat').scrollTop = $('chat').scrollHeight;
  }
  async function createCase() {
    if (busy) return;
    setBusy(true);
    status('يتم إنشاء جلسة…', true);
    try {
      const result = await api('/api/sessions', { method: 'POST', body: '{}' });
      current = { id: result.session.session_id, token: result.token, label: 'إفادة جديدة' };
      saveCases([current, ...savedCases().filter(item => item.id !== current.id)]);
      renderState(result.session);
      status('بدأت جلسة جديدة. رمز الدخول محفوظ في هذا المتصفح فقط.', true);
      $('message').focus();
    } catch (error) { status(error.message); }
    finally { setBusy(false); }
  }
  async function openCase(id) {
    const entry = savedCases().find(item => item.id === id);
    if (!entry) return;
    current = entry;
    setBusy(true);
    try {
      const result = await api(`/api/sessions/${entry.id}`);
      renderState(result.session);
      status('تم تحميل الجلسة المحفوظة.', true);
    } catch (error) { status(`${error.message} قد يكون رمز الجلسة غير موجود في هذا المتصفح.`); }
    finally { setBusy(false); }
  }
  async function sendMessage(event) {
    event.preventDefault();
    if (busy || !current || !session) return;
    const textarea = $('message');
    const text = textarea.value;
    if (!text.trim()) return;
    setBusy(true);
    status('جارٍ الحفظ…', true);
    try {
      const payload = { expected_revision: session.revision, text, kind,
        allow_external: kind === 'question' && $('model-consent').checked };
      const result = await api(`/api/sessions/${current.id}/messages`, { method: 'POST', body: JSON.stringify(payload) });
      renderState(result.session);
      textarea.value = '';
      $('model-consent').checked = false;
      status('تم حفظ الرسالة وتحديث الحالة.', true);
    } catch (error) {
      if (error.status === 409) {
        try {
          const latest = await api(`/api/sessions/${current.id}`);
          renderState(latest.session);
          status('حُفظ نصك في الحقل. تم تحديث الحالة الأحدث؛ راجعها ثم أعد الإرسال.', false);
        } catch { status('تعذر تحديث الحالة. بقي نصك في الحقل؛ لا ترسل مرة أخرى قبل إعادة تحميل الجلسة.'); }
      } else {
        try {
          const latest = await api(`/api/sessions/${current.id}`);
          renderState(latest.session);
          status(`لم يكتمل الإرسال: ${error.message} بقي نصك في الحقل. قرأنا أحدث حالة؛ راجعها قبل إعادة المحاولة.`);
        } catch { status('تعذر تأكيد حالة الحفظ بعد الخطأ. بقي نصك في الحقل؛ أعد فتح الجلسة قبل إعادة المحاولة.'); }
      }
    } finally { setBusy(false); }
  }
  async function pauseOrResume() {
    if (!current || !session || busy) return;
    if (session.stage === 'paused') {
      setBusy(true);
      try {
        const result = await api(`/api/sessions/${current.id}/resume`, { method: 'POST', body: JSON.stringify({ expected_revision: session.revision }) });
        renderState(result.session);
        status('استُكملت الجلسة من موضعها السابق.', true);
      } catch (error) { status(error.message); }
      finally { setBusy(false); }
      return;
    }
    const draft = $('message').value;
    setBusy(true);
    try {
      const result = await api(`/api/sessions/${current.id}/messages`, { method: 'POST', body: JSON.stringify({ expected_revision: session.revision, text: 'احفظ الجلسة', kind: 'answer' }) });
      renderState(result.session);
      $('message').value = draft;
      status('حُفظت الجلسة؛ يمكنك استكمالها لاحقًا.', true);
    } catch (error) { status(error.message); }
    finally { setBusy(false); }
  }
  async function confirmSummary() {
    if (!current || !session) return;
    setBusy(true);
    try {
      const result = await api(`/api/sessions/${current.id}/confirm`, { method: 'POST', body: JSON.stringify({ expected_revision: session.revision }) });
      renderState(result.session);
      status('تم تأكيد الملخص الحالي.', true);
    } catch (error) { status(error.message); }
    finally { setBusy(false); }
  }
  async function changeJurisdiction() {
    if (!current || !session) return;
    try {
      const result = await api(`/api/sessions/${current.id}/jurisdiction`, { method: 'PUT', body: JSON.stringify({ expected_revision: session.revision, jurisdiction: $('jurisdiction').value }) });
      renderState(result.session);
      status('تم تحديث الاختصاص المستخدم لفلترة المصادر.', true);
    } catch (error) { status(error.message); $('jurisdiction').value = session.jurisdiction || ''; }
  }
  async function downloadReport() {
    if (!current) return;
    try {
      const response = await fetch(`/api/sessions/${current.id}/report.md`, { headers: { 'X-Case-Token': current.token } });
      if (!response.ok) throw new Error('تعذر تنزيل التقرير.');
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `ACCP-${current.id.slice(0, 8)}-summary.md`;
      anchor.click();
      URL.revokeObjectURL(url);
      status('تم تنزيل التقرير.', true);
    } catch (error) { status(error.message); }
  }
  async function checkHealth() {
    try {
      const response = await fetch('/api/health');
      const health = await response.json();
      const configured = !!health.external_model_configured;
      $('model-consent-row').dataset.configured = String(configured);
      $('model-consent-row').classList.toggle('hidden', !(kind === 'question' && configured));
    } catch {
      $('model-consent-row').dataset.configured = 'false';
      $('model-consent-row').classList.add('hidden');
    }
  }

  $('start-case').addEventListener('click', createCase);
  $('new-case').addEventListener('click', createCase);
  $('chat-form').addEventListener('submit', sendMessage);
  $('pause-case').addEventListener('click', pauseOrResume);
  $('confirm-summary').addEventListener('click', confirmSummary);
  $('download-report').addEventListener('click', downloadReport);
  $('jurisdiction').addEventListener('change', changeJurisdiction);
  $('case-list').addEventListener('change', () => { $('resume-case').disabled = !$('case-list').value; });
  $('resume-case').addEventListener('click', () => openCase($('case-list').value));
  $('toggle-mode').addEventListener('click', () => {
    kind = kind === 'answer' ? 'question' : 'answer';
    $('mode-label').textContent = kind === 'answer' ? 'وضع المقابلة: إجابة' : 'وضع السؤال للبوت';
    $('toggle-mode').textContent = kind === 'answer' ? 'عندي سؤال للبوت' : 'العودة للإجابة';
    $('message').placeholder = kind === 'answer' ? 'اكتب إجابتك هنا…' : 'اكتب سؤالك هنا…';
    $('model-consent-row').classList.toggle('hidden', kind !== 'question' || $('model-consent-row').dataset.configured !== 'true');
  });
  renderCaseList();
  checkHealth();
  const recent = savedCases()[0];
  if (recent) openCase(recent.id);
})();
