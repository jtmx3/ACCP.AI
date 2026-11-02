(() => {
  const $ = (id) => document.getElementById(id);
  const STORAGE_KEY = 'accp.local.cases.v1';
  const labels = {
    narrative: 'وصف الواقعة', threat: 'وجود تهديد', impersonation: 'انتحال شخصية', platform: 'المنصة', details: 'تفاصيل الواقعة',
    additional_details: 'تفاصيل إضافية', publication_threat: 'تهديد بالنشر', direct_danger: 'الخطر المباشر الحالي',
    active_unauthorized_funds: 'أموال معرضة لتصرف دون إذن', fraud_description: 'وصف خداع مالي', harassment: 'مضايقة مذكورة',
    contact_repeated: 'تكرار التواصل دون استنتاج تهديد', initial_time: 'وقت البداية', latest_time: 'أحدث تواصل',
    repeated: 'تكرار التهديد أو المضايقة', requested_amount: 'المبلغ المطلوب', paid_amount: 'المبلغ المدفوع',
    payment_method: 'وسيلة الدفع', deadline: 'مهلة الدفع', published: 'حدوث النشر', identity: 'هوية الشخص',
    contact: 'معرّف التواصل', evidence: 'الأدلة المذكورة', account_access: 'دخول غير مصرح به'
  };
  let active = null; // { id, token, label, generation, session }
  let activeGeneration = 0;
  let kind = 'answer';
  let externalConfigured = false;
  let creating = false;
  const caseCache = new Map();
  const caseOps = new Map();
  const requestSequence = new Map();
  const appliedSequence = new Map();

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
    const selected = active?.id || list.value;
    list.replaceChildren(new Option('اختر جلسة محفوظة', ''));
    for (const item of savedCases()) list.add(new Option(`${item.label || 'إفادة'} · ${item.id.slice(0, 8)}`, item.id));
    if (selected && savedCases().some(item => item.id === selected)) list.value = selected;
    $('resume-case').disabled = !list.value;
  }
  function setStatus(text = '', ok = false, ctx = null) {
    if (ctx && !isActive(ctx)) return;
    const element = $('status-message');
    element.textContent = text;
    element.style.color = ok ? '#075a56' : '';
  }
  function isActive(ctx) {
    return !!ctx && !!active && active.id === ctx.id && active.generation === ctx.generation && activeGeneration === ctx.generation;
  }
  function snapshotActive() {
    if (!active || !active.session) return null;
    return Object.freeze({ id: active.id, token: active.token, label: active.label, generation: active.generation,
      revision: active.session.revision, sessionId: active.session.session_id });
  }
  function updateControls() {
    const id = active?.id;
    const locked = !!id && caseOps.has(id);
    const ready = !!(active?.session && active.session.session_id === active.id);
    $('send').disabled = !ready || locked;
    $('pause-case').disabled = !ready || locked;
    $('confirm-summary').disabled = !ready || locked || !!active?.session?.confirmed;
    $('jurisdiction').disabled = !ready || locked;
    $('resume-case').disabled = !$('case-list').value;
    $('start-case').disabled = creating;
    $('new-case').disabled = creating;
  }
  function nextRequest(ctx) {
    const value = (requestSequence.get(ctx.id) || 0) + 1;
    requestSequence.set(ctx.id, value);
    return value;
  }
  function requestIsLatest(ctx, sequence) {
    return sequence >= (appliedSequence.get(ctx.id) || 0);
  }
  function cacheState(state) {
    if (!state || !state.session_id || !Number.isInteger(state.revision)) return;
    const old = caseCache.get(state.session_id);
    if (!old || state.revision > old.revision) caseCache.set(state.session_id, state);
    else if (state.revision === old.revision) caseCache.set(state.session_id, state);
  }
  function commitState(ctx, state, sequence) {
    if (!ctx || !state || state.session_id !== ctx.id) return false;
    const cached = caseCache.get(ctx.id);
    if (cached && state.revision < cached.revision) {
      if (isActive(ctx) && active.session == null) {
        active.session = cached;
        renderState(ctx, cached);
      }
      return false;
    }
    if (!requestIsLatest(ctx, sequence) && (!cached || state.revision <= cached.revision)) return false;
    cacheState(state);
    appliedSequence.set(ctx.id, Math.max(sequence, appliedSequence.get(ctx.id) || 0));
    if (!isActive(ctx)) return false;
    const freshest = caseCache.get(ctx.id) || state;
    active.session = freshest;
    renderState(ctx, freshest);
    updateControls();
    return true;
  }
  async function api(path, options = {}, ctx = null) {
    const headers = { ...(options.headers || {}) };
    if (options.body != null && !headers['Content-Type']) headers['Content-Type'] = 'application/json';
    if (ctx?.token) headers['X-Case-Token'] = ctx.token;
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
  function beginMutation(ctx) {
    if (!ctx || caseOps.has(ctx.id)) return null;
    const owner = Symbol(ctx.id);
    caseOps.set(ctx.id, owner);
    updateControls();
    return owner;
  }
  function endMutation(ctx, owner) {
    if (ctx && caseOps.get(ctx.id) === owner) caseOps.delete(ctx.id);
    updateControls();
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
        const item = document.createElement('div');
        item.className = 'citation-item';
        const anchor = document.createElement('a');
        anchor.href = citation.url;
        anchor.target = '_blank';
        anchor.rel = 'noopener noreferrer';
        anchor.textContent = `${citation.publisher}: ${citation.title}`;
        item.append(anchor);
        if (citation.excerpt) {
          const excerpt = document.createElement('div');
          excerpt.className = 'citation-excerpt';
          excerpt.textContent = `المقتطف المستخدم: «${citation.excerpt}»`;
          item.append(excerpt);
        }
        links.append(item);
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
    const rule = assessment.priority_rule ? `\nقاعدة الدرجة: ${assessment.priority_rule}؛ الوقائع الداعمة: ${(assessment.supporting_fields || []).map(key => labels[key] || key).join('، ')}` : '';
    const relevant = (assessment.relevant_fields || []).map(key => labels[key] || key).join('، ');
    $('assessment-reasons').textContent = `${assessment.reasons?.length ? `الأسباب: ${assessment.reasons.join('؛ ')}` : 'لم تتوفر إشارة كافية لتحديد الأولوية.'}${rule}\nالحقول ذات الصلة: ${relevant || 'وصف الواقعة فقط'}\nاكتمال المعلومات: ${assessment.completeness_percent ?? 0}% — ${assessment.completeness_note || ''}\nجودة الاستخراج: ${assessment.extraction_quality || 'غير متاحة'}؛ لا يوجد تحقق مستقل.`;
    const factsList = $('facts-list');
    factsList.replaceChildren();
    const facts = state.facts || {};
    const order = ['narrative', 'threat', 'impersonation', 'platform', 'details', 'additional_details', 'publication_threat', 'direct_danger', 'active_unauthorized_funds', 'fraud_description', 'harassment', 'contact_repeated', 'initial_time', 'latest_time', 'repeated', 'requested_amount', 'paid_amount', 'payment_method', 'deadline', 'published', 'identity', 'contact', 'evidence', 'account_access'];
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
    $('confirm-summary').disabled = !!state.confirmed || caseOps.has(state.session_id);
  }
  function renderState(ctx, state) {
    if (!isActive(ctx) || !state || state.session_id !== ctx.id) return false;
    active.session = state;
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
    updateControls();
    return true;
  }
  function makeContext(entry, generation) {
    return Object.freeze({ id: entry.id, token: entry.token, label: entry.label || 'إفادة', generation });
  }
  async function createCase() {
    if (creating) return;
    creating = true;
    const generationAtStart = activeGeneration;
    updateControls();
    setStatus('يتم إنشاء جلسة…', true);
    try {
      const result = await api('/api/sessions', { method: 'POST', body: '{}' });
      const entry = { id: result.session.session_id, token: result.token, label: 'إفادة جديدة' };
      saveCases([entry, ...savedCases().filter(item => item.id !== entry.id)]);
      cacheState(result.session);
      if (activeGeneration === generationAtStart) {
        $('message').value = '';
        await openCase(entry.id);
        setStatus('بدأت جلسة جديدة. رمز الدخول محفوظ في هذا المتصفح فقط.', true, active);
        $('message').focus();
      }
    } catch (error) {
      if (activeGeneration === generationAtStart) setStatus(error.message);
    } finally {
      creating = false;
      updateControls();
    }
  }
  async function openCase(id) {
    const entry = savedCases().find(item => item.id === id);
    if (!entry) return;
    const ctx = makeContext(entry, ++activeGeneration);
    active = { ...ctx, session: null };
    $('case-list').value = id;
    $('welcome').classList.add('hidden');
    $('case-view').classList.remove('hidden');
    $('case-title').textContent = `الحالة ${id.slice(0, 8)}`;
    $('stage-badge').textContent = 'جارٍ التحميل';
    $('chat').replaceChildren();
    $('review').classList.add('hidden');
    const cached = caseCache.get(id);
    if (cached) {
      active.session = cached;
      renderState(ctx, cached);
    }
    updateControls();
    const sequence = nextRequest(ctx);
    try {
      const result = await api(`/api/sessions/${ctx.id}`, {}, ctx);
      commitState(ctx, result.session, sequence);
      setStatus('تم تحميل الجلسة المحفوظة.', true, ctx);
    } catch (error) {
      if (isActive(ctx)) setStatus(`${error.message} قد يكون رمز الجلسة غير موجود في هذا المتصفح.`);
    }
  }
  async function refreshCase(ctx) {
    const sequence = nextRequest(ctx);
    try {
      const latest = await api(`/api/sessions/${ctx.id}`, {}, ctx);
      commitState(ctx, latest.session, sequence);
      return latest.session;
    } catch { return null; }
  }
  async function sendMessage(event) {
    event.preventDefault();
    const ctx = snapshotActive();
    if (!ctx) return;
    const owner = beginMutation(ctx);
    if (!owner) return;
    const textarea = $('message');
    const text = textarea.value;
    if (!text.trim()) { endMutation(ctx, owner); return; }
    const consent = $('model-consent').checked;
    const sequence = nextRequest(ctx);
    setStatus('جارٍ الحفظ…', true, ctx);
    try {
      const payload = { expected_revision: ctx.revision, text, kind,
        allow_external: kind === 'question' && consent,
        allow_external_extraction: kind === 'answer' && consent };
      const result = await api(`/api/sessions/${ctx.id}/messages`, { method: 'POST', body: JSON.stringify(payload) }, ctx);
      const rendered = commitState(ctx, result.session, sequence);
      if (rendered && $('message').value === text) $('message').value = '';
      if (rendered) {
        if (result.external_data_sent) {
          const purpose = kind === 'question' ? 'نص السؤال الحالي ومقتطفات المصادر المختارة' : 'نص الرسالة الحالية فقط لاستخراج حقائق منظمة';
          setStatus(result.external_model_used ? `أُرسل ${purpose} لمزود خارجي بموافقتك، واستُخدمت نتيجة موثقة.` : `أُرسل ${purpose} لمزود خارجي بموافقتك؛ لم تُستخدم نتيجة صالحة، فاستُخدمت القواعد/الإجابة المحلية.`, true, ctx);
        } else setStatus('تم حفظ الرسالة وتحديث الحالة.', true, ctx);
      }
    } catch (error) {
      if (error.status === 409) {
        await refreshCase(ctx);
        setStatus('تغيّرت الحالة. حدّثنا النسخة الأحدث؛ بقي نصك في الحقل ولم نعد الإرسال تلقائيًا.', false, ctx);
      } else {
        await refreshCase(ctx);
        setStatus(`تعذر تأكيد نتيجة الحفظ: ${error.message} بقي نصك في الحقل؛ راجع الحالة الأحدث قبل إعادة المحاولة.`, false, ctx);
      }
    } finally {
      $('model-consent').checked = false;
      endMutation(ctx, owner);
    }
  }
  async function pauseOrResume() {
    const ctx = snapshotActive();
    if (!ctx) return;
    const owner = beginMutation(ctx);
    if (!owner) return;
    const sequence = nextRequest(ctx);
    const draft = $('message').value;
    try {
      if (active.session.stage === 'paused') {
        const result = await api(`/api/sessions/${ctx.id}/resume`, { method: 'POST', body: JSON.stringify({ expected_revision: ctx.revision }) }, ctx);
        commitState(ctx, result.session, sequence);
        setStatus('استُكملت الجلسة من موضعها السابق.', true, ctx);
      } else {
        const result = await api(`/api/sessions/${ctx.id}/messages`, { method: 'POST', body: JSON.stringify({ expected_revision: ctx.revision, text: 'احفظ الجلسة', kind: 'answer' }) }, ctx);
        commitState(ctx, result.session, sequence);
        if (isActive(ctx)) $('message').value = draft;
        setStatus('حُفظت الجلسة؛ يمكنك استكمالها لاحقًا.', true, ctx);
      }
    } catch (error) {
      if (error.status === 409) await refreshCase(ctx);
      setStatus(error.message, false, ctx);
    } finally { endMutation(ctx, owner); }
  }
  async function confirmSummary() {
    const ctx = snapshotActive();
    if (!ctx) return;
    const owner = beginMutation(ctx);
    if (!owner) return;
    const sequence = nextRequest(ctx);
    try {
      const result = await api(`/api/sessions/${ctx.id}/confirm`, { method: 'POST', body: JSON.stringify({ expected_revision: ctx.revision }) }, ctx);
      commitState(ctx, result.session, sequence);
      setStatus('تم تأكيد الملخص الحالي.', true, ctx);
    } catch (error) {
      if (error.status === 409) await refreshCase(ctx);
      setStatus(error.message, false, ctx);
    } finally { endMutation(ctx, owner); }
  }
  async function changeJurisdiction() {
    const ctx = snapshotActive();
    if (!ctx) return;
    const owner = beginMutation(ctx);
    if (!owner) return;
    const jurisdiction = $('jurisdiction').value;
    const sequence = nextRequest(ctx);
    try {
      const result = await api(`/api/sessions/${ctx.id}/jurisdiction`, { method: 'PUT', body: JSON.stringify({ expected_revision: ctx.revision, jurisdiction }) }, ctx);
      commitState(ctx, result.session, sequence);
      setStatus('تم تحديث الاختصاص المستخدم لفلترة المصادر.', true, ctx);
    } catch (error) {
      if (error.status === 409) await refreshCase(ctx);
      if (isActive(ctx)) $('jurisdiction').value = active.session?.jurisdiction || '';
      setStatus(error.message, false, ctx);
    } finally { endMutation(ctx, owner); }
  }
  async function downloadReport() {
    const ctx = snapshotActive();
    if (!ctx) return;
    try {
      const response = await fetch(`/api/sessions/${ctx.id}/report.md`, { headers: { 'X-Case-Token': ctx.token } });
      if (!response.ok) throw new Error('تعذر تنزيل التقرير.');
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `ACCP-${ctx.id.slice(0, 8)}-summary.md`;
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setStatus('تم تنزيل التقرير.', true, ctx);
    } catch (error) { setStatus(error.message, false, ctx); }
  }
  async function loadMetadata() {
    try {
      const result = await api('/api/metadata');
      const select = $('jurisdiction');
      const oldValue = select.value;
      select.replaceChildren(new Option('غير محدد', ''));
      for (const item of result.jurisdictions || []) select.add(new Option(item.label, item.code));
      if ([...select.options].some(option => option.value === oldValue)) select.value = oldValue;
    } catch { /* If metadata is unavailable, only the empty/unscoped option remains. */ }
  }
  async function checkHealth() {
    try {
      const health = await api('/api/health');
      externalConfigured = !!health.external_model_configured;
    } catch { externalConfigured = false; }
    $('model-consent-row').dataset.configured = String(externalConfigured);
    updateConsentVisibility();
  }
  function updateConsentVisibility() {
    $('model-consent-row').classList.toggle('hidden', !externalConfigured);
    $('model-consent').checked = false;
    $('model-consent-copy').textContent = kind === 'question'
      ? 'أوافق على إرسال نص هذا السؤال الحالي ومقتطفات المصادر المختارة فقط إلى مزود خارجي لهذا الرد. لن يُرسل سجل المقابلة.'
      : 'أوافق على إرسال نص رسالتي الحالية فقط لاستخراج حقائق منظمة. قد يتضمن النص بيانات شخصية؛ لن يُرسل سجل المحادثة أو إجابات سابقة.';
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
    updateConsentVisibility();
  });
  renderCaseList();
  updateControls();
  loadMetadata();
  checkHealth();
  const recent = savedCases()[0];
  if (recent) openCase(recent.id);
})();
