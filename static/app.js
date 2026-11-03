(() => {
  const $ = (id) => document.getElementById(id);
  const STORAGE_KEY = 'accp.local.cases.v1';
  const labels = {
    narrative: 'وصف الواقعة', threat: 'وجود تهديد', impersonation: 'انتحال شخصية', platform: 'المنصة', details: 'تفاصيل الواقعة',
    unclassified_details: 'توضيح واقعة غير مصنفة', incident_type_review: 'مراجعة تصنيف الواقعة',
    additional_details: 'تفاصيل إضافية', publication_threat: 'تهديد بالنشر', direct_danger: 'الخطر المباشر الحالي',
    active_unauthorized_funds: 'أموال معرضة لتصرف دون إذن', fraud_description: 'وصف خداع مالي', harassment: 'مضايقة مذكورة',
    contact_repeated: 'تكرار التواصل دون استنتاج تهديد', initial_time: 'وقت البداية', latest_time: 'أحدث تواصل',
    repeated: 'تكرار التهديد أو المضايقة', requested_amount: 'المبلغ المطلوب', paid_amount: 'المبلغ المدفوع',
    payment_method: 'وسيلة الدفع', deadline: 'مهلة الدفع', published: 'حدوث النشر', identity: 'هوية الشخص',
    contact: 'معرّف التواصل', contact_identifier_disclosure: 'الإفصاح عن معرّف التواصل',
    evidence: 'الأدلة المذكورة', account_access: 'دخول غير مصرح به',
    requested_amount_relation: 'علاقة المبلغ المطلوب الجديد بالسابق', paid_amount_relation: 'علاقة مبلغ الدفع الجديد بالسابق'
  };
  const factStatusLabels = {
    reported: 'مذكورة بحسب الإفادة', denied: 'منفية صراحة بحسب الإفادة', unknown: 'غير معروفة للمستخدم',
    not_remembered: 'لا يتذكرها المستخدم', declined: 'فضّل عدم الإجابة',
    unresolved: 'غير محسومة بعد الاستيضاح', conflict: 'تعارض فعلي يحتاج إلى توضيح'
  };
  let active = null; // { id, token, label, generation, session }
  let activeGeneration = 0;
  let kind = 'answer';
  let externalConfigured = false;
  let providerConfig = null;
  let providerProbe = null;
  let providerGenerationTest = false;
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
  function formatTime(value) {
    if (!value) return 'غير متاح';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString('ar');
  }
  function providerLabel(provider) {
    if (provider === 'ollama') return 'Ollama محلي';
    if (provider === 'openai-compatible') return 'مزود متوافق خارجي';
    return 'القواعد المحلية';
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
    panel.dataset.summaryVersion = String(state.summary_version ?? '');
    panel.dataset.confirmed = String(!!state.confirmed);
    const assessment = state.assessment || {};
    $('priority-chip').textContent = `أولوية المراجعة: ${assessment.priority_score == null ? 'غير محددة' : `${assessment.priority_score}/100`} · ${assessment.priority_label || ''}`;
    const rule = assessment.priority_rule ? `\nقاعدة الدرجة: ${assessment.priority_rule}؛ الوقائع الداعمة: ${(assessment.supporting_fields || []).map(key => labels[key] || 'معلومة إضافية').join('، ')}` : '';
    const relevant = (assessment.relevant_fields || []).map(key => labels[key] || 'معلومة إضافية').join('، ');
    $('assessment-reasons').textContent = `${assessment.reasons?.length ? `الأسباب: ${assessment.reasons.join('؛ ')}` : 'لم تتوفر إشارة كافية لتحديد الأولوية.'}${rule}\nالحقول ذات الصلة: ${relevant || 'وصف الواقعة فقط'}\nاكتمال المعلومات: ${assessment.completeness_percent ?? 0}% — ${assessment.completeness_note || ''}\nجودة الاستخراج: ${assessment.extraction_quality || 'غير متاحة'}؛ لا يوجد تحقق مستقل.`;
    const factsList = $('facts-list');
    factsList.replaceChildren();
    const facts = state.facts || {};
    const messagesById = new Map((state.messages || []).map(message => [message.id, message]));
    const order = ['narrative', 'threat', 'impersonation', 'platform', 'details', 'additional_details', 'publication_threat', 'direct_danger', 'active_unauthorized_funds', 'fraud_description', 'harassment', 'contact_repeated', 'initial_time', 'latest_time', 'repeated', 'requested_amount', 'paid_amount', 'payment_method', 'deadline', 'published', 'identity', 'contact', 'contact_identifier_disclosure', 'evidence', 'account_access'];
    for (const key of order) {
      const fact = facts[key];
      if (!fact) continue;
      const wrapper = document.createElement('div');
      wrapper.className = 'fact-item';
      const term = document.createElement('dt');
      term.textContent = labels[key] || 'معلومة إضافية';
      const description = document.createElement('dd');
      description.className = 'fact-value';
      let value = fact.status === 'reported' ? fact.value : null;
      if (typeof value === 'boolean') value = value ? 'نعم بحسب الإفادة' : 'لا بحسب الإفادة';
      description.textContent = `القيمة: ${value == null ? 'لا توجد قيمة مثبتة لهذا الحقل' : String(value)}`;
      const status = document.createElement('div');
      status.className = `fact-status status-${fact.status || 'unknown'}`;
      status.textContent = `الحالة: ${factStatusLabels[fact.status] || 'غير متاحة'}`;
      wrapper.append(term, status, description);
      if (fact.quote) {
        const quote = document.createElement('div');
        quote.className = 'fact-quote';
        quote.textContent = `الاقتباس: «${fact.quote}»`;
        wrapper.append(quote);
      }
      if (key === 'details') {
        const requested = facts.requested_amount;
        const previous = [...(requested?.history || [])].reverse().find(item =>
          ['explicit_correction', 'explicit_amount_relation_correction'].includes(item.reason) && item.value != null);
        const currentDigits = String(requested?.value ?? '').replace(/[٠-٩]/g, digit => String('٠١٢٣٤٥٦٧٨٩'.indexOf(digit))).replace(/\D/g, '');
        const previousDigits = String(previous?.value ?? '').replace(/[٠-٩]/g, digit => String('٠١٢٣٤٥٦٧٨٩'.indexOf(digit))).replace(/\D/g, '');
        const detailDigits = String(fact.quote || fact.value || '').replace(/[٠-٩]/g, digit => String('٠١٢٣٤٥٦٧٨٩'.indexOf(digit))).replace(/\D/g, '');
        const correctionRecorded = (state.money_events || []).some(event => event.kind === 'requested' && event.relation === 'correction' &&
          event.message_id === requested?.message_id && String(event.amount || '').replace(/[٠-٩]/g, digit => String('٠١٢٣٤٥٦٧٨٩'.indexOf(digit))).replace(/\D/g, '') === currentDigits);
        if (previous && previousDigits && currentDigits && previousDigits !== currentDigits && detailDigits.includes(previousDigits) && correctionRecorded) {
          const note = document.createElement('div');
          note.className = 'fact-correction-note';
          note.textContent = `توضيح مستند إلى تصحيح لاحق: المبلغ المطلوب المعتمد حاليًا ${requested.value}؛ وكانت الإفادة الأصلية في هذا الموضع تذكر ${previous.value}. بقي الاقتباس الأصلي أعلاه دون تعديل.`;
          wrapper.append(note);
        }
      }
      const sourceMessage = messagesById.get(fact.message_id);
      const sourceTime = document.createElement('div');
      sourceTime.className = 'fact-time';
      sourceTime.textContent = `وقت الرسالة الأصلية: ${formatTime(sourceMessage?.created_at || fact.source_message_at)}`;
      wrapper.append(sourceTime);
      const updateTime = document.createElement('div');
      updateTime.className = 'fact-time';
      updateTime.textContent = `وقت آخر تحديث للمعلومة: ${formatTime(fact.updated_at || fact.last_updated_at)}`;
      wrapper.append(updateTime);
      for (const alternative of fact.alternatives || []) {
        const item = document.createElement('div');
        item.className = 'fact-alternative';
        const altStatus = factStatusLabels[alternative.status] || 'حالة غير متاحة';
        item.textContent = `إفادة بديلة (${altStatus}): ${alternative.value ?? 'لا توجد قيمة'} — «${alternative.quote || ''}»`;
        wrapper.append(item);
      }
      for (const previous of fact.history || []) {
        if (!previous.quote) continue;
        const item = document.createElement('div');
        item.className = 'fact-quote';
        item.textContent = `إفادة سابقة (${factStatusLabels[previous.status] || 'حالة غير متاحة'}): ${previous.value ?? 'لا توجد قيمة'} — «${previous.quote}»`;
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
    const gapPanel = $('information-gaps');
    gapPanel.replaceChildren();
    const relationSlots = new Set(['requested_amount_relation', 'paid_amount_relation']);
    const missingItems = (assessment.relevant_fields || [])
      .filter(key => !facts[key] && !relationSlots.has(key)).map(key => labels[key] || 'معلومة إضافية');
    const unknownItems = Object.entries(facts)
      .filter(([, fact]) => ['unknown', 'not_remembered', 'declined'].includes(fact.status))
      .map(([key, fact]) => `${labels[key] || 'معلومة إضافية'} — ${factStatusLabels[fact.status]}`);
    const unresolvedItems = Object.entries(facts)
      .filter(([, fact]) => fact.status === 'unresolved')
      .map(([key]) => labels[key] || 'معلومة إضافية');
    const unresolvedMoney = (state.money_events || [])
      .filter(event => ['unresolved', 'unresolved_after_clarification'].includes(event.relation));
    for (const event of unresolvedMoney) {
      const kind = event.kind === 'paid' ? 'مدفوع' : 'مطلوب';
      unresolvedItems.push(`علاقة المبلغ ${kind} (${event.amount || 'غير محدد'}) تحتاج توضيحًا`);
    }
    const conflicts = Object.entries(facts)
      .filter(([, fact]) => fact.status === 'conflict')
      .map(([key, fact]) => `${labels[key] || 'معلومة إضافية'} — ${fact.alternatives?.length || 0} إفادة بديلة محفوظة`);
    const gapGroups = [
      ['لم تُسجّل بعد', missingItems],
      ['غير معروفة أو لم يتذكرها المستخدم أو فضّل عدم الإجابة', unknownItems],
      ['تحتاج إلى توضيح', unresolvedItems],
      ['تعارضات فعلية', conflicts],
    ];
    let hasGaps = false;
    for (const [title, items] of gapGroups) {
      if (!items.length) continue;
      hasGaps = true;
      const group = document.createElement('section');
      group.className = 'gap-group';
      if (title === 'تعارضات فعلية' || title === 'تحتاج إلى توضيح') group.classList.add('gap-attention');
      const heading = document.createElement('h3');
      heading.textContent = title;
      const list = document.createElement('ul');
      for (const text of items) {
        const item = document.createElement('li');
        item.textContent = text;
        list.append(item);
      }
      group.append(heading, list);
      gapPanel.append(group);
    }
    if (!hasGaps) gapPanel.textContent = 'لا توجد معلومات مجهولة أو غير محسومة أو متعارضة في الحقول ذات الصلة.';
    $('confirmation-state').textContent = `إصدار الملخص المعروض: ${state.summary_version ?? 'غير متاح'} · ${state.confirmed ? 'مؤكد لهذا الإصدار' : 'غير مؤكد'}. ${state.confirmed ? 'أي تعديل لاحق يتطلب مراجعة جديدة.' : ''}`;
    $('confirm-summary').disabled = !!state.confirmed || caseOps.has(state.session_id) || unresolvedMoney.length > 0;
  }
  function renderState(ctx, state) {
    if (!isActive(ctx) || !state || state.session_id !== ctx.id) return false;
    active.session = state;
    $('welcome').classList.add('hidden');
    $('case-view').classList.remove('hidden');
    $('case-title').textContent = `الحالة ${state.session_id.slice(0, 8)}`;
    $('case-updated-at').textContent = `آخر تحديث للحالة: ${formatTime(state.updated_at)}`;
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
          const provider = providerLabel(result.provider_call?.provider || providerConfig?.provider);
          const purpose = result.external_model_purpose === 'grounded_question'
            ? 'نص السؤال الحالي ومقتطفات المصادر المختارة' : 'نص الرسالة الحالية فقط';
          const extractionOutcome = result.extraction_outcome || result.provider_call?.extraction_outcome;
          const extractionCall = result.external_model_purpose === 'current_message_extraction';
          const latency = result.provider_call?.latency_ms ?? 'غير متاح';
          if (extractionCall && extractionOutcome === 'valid_empty') {
            $('runtime-mode').textContent = `القواعد المحلية — رد صالح من ${provider} بلا وقائع مستخرجة.`;
            setStatus(`رد ${provider} صالح بنيويًا لكنه لم يتضمن وقائع؛ لم يُعد ذلك نجاحًا للاستخراج، وطُبقت القواعد المحلية على الرسالة.`, false, ctx);
            $('runtime-detail').textContent = `آخر عملية: رد صالح بلا مقترحات حقائق؛ القواعد المحلية عالجت الرسالة. زمن الاستجابة ${latency} مللي ثانية.`;
          } else if (extractionCall && extractionOutcome === 'all_facts_filtered') {
            const reasonLabels = { quote_mismatch: 'اقتباس غير مطابق', value_inconsistent: 'قيمة غير متسقة',
              unsupported_status_evidence: 'حالة بلا دليل كافٍ', other_validation: 'تحقق آخر' };
            const reasonCounts = result.provider_call?.filtered_fact_code_counts || {};
            const reasons = Object.entries(reasonCounts).map(([code, count]) => `${reasonLabels[code] || 'سبب تحقق آخر'}: ${count}`).join('، ');
            $('runtime-mode').textContent = `القواعد المحلية — استُبعدت جميع مقترحات ${provider}.`;
            setStatus(`أعاد ${provider} مقترحات، لكن المدقق استبعدها جميعًا${reasons ? ` (${reasons})` : ''}؛ طُبقت القواعد المحلية.`, false, ctx);
            $('runtime-detail').textContent = `آخر عملية: استُبعدت جميع المقترحات (${result.provider_call?.filtered_fact_count || 0})؛ ${reasons || 'أسباب التصفية مسجلة في التشخيص'}؛ زمن الاستجابة ${latency} مللي ثانية.`;
          } else if (result.external_model_result_accepted) {
            const filteredCount = Number(result.provider_call?.filtered_fact_count || 0);
            const runtimeLabel = result.runtime_mode === 'local_model' ? 'نموذج Ollama المحلي' : 'مزود خارجي';
            $('runtime-mode').textContent = `${runtimeLabel} — استُخدم في آخر إرسال بموافقتك.`;
            const filterNote = filteredCount ? `؛ واستبعد النظام ${filteredCount} من الحقائق غير المسندة عن الحالة` : '';
            const extractedCount = Number(result.provider_call?.fact_fields?.length || 0);
            const acceptedSummary = extractionCall ? `عدد الوقائع المقبولة: ${extractedCount}` : 'قُبل جواب الاستناد إلى المصادر';
            setStatus(`أُرسل ${purpose} بموافقتك إلى ${provider}؛ ${acceptedSummary}${filterNote}.`, true, ctx);
            $('runtime-detail').textContent = filteredCount
              ? `آخر عملية: ${extractionCall ? `استخراج من ${provider}؛ عدد الوقائع المقبولة ${extractedCount}، وعدد المقترحات المصفاة ${filteredCount}` : `قُبل الجواب الموثق من ${provider}`}؛ زمن الاستجابة ${latency} مللي ثانية.`
              : `آخر عملية: ${extractionCall ? `استخراج من ${provider}؛ عدد الوقائع المقبولة ${extractedCount}` : `قُبل الجواب الموثق من ${provider}`}؛ زمن الاستجابة ${latency} مللي ثانية.`;
          } else {
            $('runtime-mode').textContent = `القواعد المحلية — عاد الطلب إليها بعد تعذر قبول نتيجة ${provider}.`;
            setStatus(`لم تُقبل نتيجة ${provider} بعد إرسال ${purpose} بموافقتك؛ فعاد الطلب إلى ${result.fallback_to === 'local_rag' ? 'الإجابة المحلية المستندة إلى المصادر' : 'القواعد المحلية'}.`, false, ctx);
            $('runtime-detail').textContent = `آخر عملية: فشل أو رُفضت نتيجة ${provider}؛ الرجوع إلى ${result.fallback_to === 'local_rag' ? 'الإجابة المحلية' : 'القواعد المحلية'}.`;
          }
        } else if (result.question_detected) {
          $('runtime-mode').textContent = 'القواعد المحلية — أُجيب عن السؤال داخل وضع المقابلة.';
          setStatus('فهمنا الرسالة كسؤال وأجبنا محليًا دون تبديل الوضع أو إرسال نصها؛ بقي السؤال المعلّق كما هو.', true, ctx);
          $('runtime-detail').textContent = 'آخر عملية: إجابة سؤال داخل وضع المقابلة محليًا؛ لم تُرسل بيانات الحالة إلى مزود.';
        } else {
          $('runtime-mode').textContent = 'القواعد المحلية — آخر رسالة عولجت دون إرسالها إلى نموذج.';
          setStatus('تم حفظ الرسالة وتحديث الحالة بالقواعد المحلية.', true, ctx);
          $('runtime-detail').textContent = 'آخر عملية: القواعد المحلية؛ لم يُرسل نص الرسالة إلى نموذج.';
        }
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
  async function downloadReport(format = 'md') {
    const ctx = snapshotActive();
    if (!ctx) return;
    try {
      const latest = await refreshCase(ctx);
      if (!latest || !isActive(ctx)) throw new Error('تعذر تحديث النسخة المعروضة قبل التنزيل.');
      const extension = format === 'json' ? 'json' : 'md';
      const response = await fetch(`/api/sessions/${ctx.id}/report.${extension}`, { headers: { 'X-Case-Token': ctx.token } });
      if (!response.ok) throw new Error('تعذر تنزيل التقرير.');
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `ACCP-${ctx.id.slice(0, 8)}-summary.${extension}`;
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setStatus(`تم تنزيل تقرير ${extension.toUpperCase()} للنسخة المعروضة (${latest.summary_version}) وحالة تأكيدها.`, true, ctx);
    } catch (error) { setStatus(error.message, false, ctx); }
  }
  async function printReport() {
    const ctx = snapshotActive();
    if (!ctx) return;
    try {
      const latest = await refreshCase(ctx);
      if (!latest || !isActive(ctx) || !['review', 'urgent_stop'].includes(latest.stage)) {
        throw new Error('تعذر تحديث ملخص قابل للطباعة.');
      }
      document.body.classList.add('print-report');
      window.print();
      window.addEventListener('afterprint', () => document.body.classList.remove('print-report'), { once: true });
      setTimeout(() => document.body.classList.remove('print-report'), 3000);
      setStatus(`الطباعة للنسخة المعروضة (${latest.summary_version}) وحالة تأكيدها.`, true, ctx);
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
  function renderProviderConfig(config) {
    providerConfig = config || null;
    externalConfigured = !!providerConfig?.configuration_complete;
    providerProbe = null;
    providerGenerationTest = false;
    $('test-provider-generation').disabled = true;
    if (!externalConfigured) {
      const providerWasSelected = providerConfig && providerConfig.provider !== 'rules';
      $('runtime-mode').textContent = providerWasSelected
        ? `القواعد المحلية — إعداد ${providerLabel(providerConfig.provider)} غير مكتمل.`
        : 'القواعد المحلية — لم يُضبط مزود نموذج.';
      $('runtime-detail').textContent = providerWasSelected
        ? 'لم يبدأ أي اتصال. يلزم استكمال اسم النموذج والإعدادات المطلوبة؛ وجود متغيرات وحده لا يثبت توفر الخدمة.'
        : 'لم يُفحص اتصال خدمة نموذج، ولا يُجرى أي استدعاء في الوضع الافتراضي.';
    } else {
      const reasoning = providerConfig.local
        ? `؛ إعداد التفكير الخاص بـOllama: ${providerConfig.ollama_reasoning_effort || 'none'}${(providerConfig.ollama_reasoning_effort || 'none') === 'none' ? ' (معطّل)' : ''}` : '';
      $('runtime-mode').textContent = `إعداد ${providerLabel(providerConfig.provider)} · النموذج: ${providerConfig.model} — لم يُثبت الاتصال بعد.`;
      $('runtime-detail').textContent = `الإعداد موجود فقط؛ لم يُفحص اتصال الخدمة أو وجود النموذج أو نجاح التوليد.${reasoning}`;
    }
    $('model-consent-row').dataset.configured = String(externalConfigured);
    updateConsentVisibility();
  }
  function renderProviderProbe(probe) {
    providerProbe = probe || null;
    providerGenerationTest = false;
    $('test-provider-generation').disabled = !(probe?.configured && probe?.service_reachable && probe?.model_available);
    if (!probe?.configured) {
      $('runtime-mode').textContent = 'القواعد المحلية — لا يوجد إعداد نموذج مكتمل.';
      $('runtime-detail').textContent = 'الإعداد غير مكتمل؛ لم يحدث اتصال بخدمة نموذج.';
    } else if (!probe.service_reachable) {
      $('runtime-mode').textContent = `القواعد المحلية — خدمة ${providerLabel(probe.provider)} غير متاحة.`;
      $('runtime-detail').textContent = `الإعداد موجود، لكن الخدمة غير متاحة (${probe.failure_code || 'سبب غير محدد'}). سيبقى التشغيل بالقواعد المحلية.`;
    } else if (!probe.model_available) {
      $('runtime-mode').textContent = `القواعد المحلية — النموذج المحدد لدى ${providerLabel(probe.provider)} غير متاح.`;
      $('runtime-detail').textContent = `الخدمة متاحة، لكن النموذج المحدد غير موجود في قائمتها (${probe.failure_code || 'تحقق من اسم النموذج'}).`;
    } else {
      $('runtime-mode').textContent = `الخدمة والنموذج متاحان لدى ${providerLabel(probe.provider)}؛ لم ينجح التوليد بعد.`;
      const reasoning = providerConfig?.local
        ? ` إعداد التفكير لـOllama: ${providerConfig.ollama_reasoning_effort || 'none'}${(providerConfig.ollama_reasoning_effort || 'none') === 'none' ? ' (معطّل)' : ''}.` : '';
      $('runtime-detail').textContent = `الخدمة متاحة والنموذج المحدد موجود (${probe.models_count} نموذجًا بالقائمة). لم يُختبر التوليد بعد.${reasoning}`;
    }
  }
  async function checkProvider() {
    $('check-provider').disabled = true;
    $('runtime-detail').textContent = 'يجري فحص الإعداد واتصال قائمة النماذج دون إرسال نص حالة أو تشغيل توليد…';
    try {
      const result = await api('/api/provider/status');
      renderProviderConfig(result.config);
      renderProviderProbe(result.probe);
      setStatus(result.probe?.service_reachable ? 'اكتمل فحص الاتصال غير التوليدي.' : 'لم يثبت اتصال خدمة النموذج؛ التشغيل بالقواعد المحلية.', !!result.probe?.service_reachable);
    } catch (error) {
      $('runtime-detail').textContent = `تعذر إجراء فحص الاتصال: ${error.message}`;
      setStatus('فشل فحص حالة مزود النموذج.', false);
    } finally { $('check-provider').disabled = false; }
  }
  async function testProviderGeneration() {
    if (!$('test-provider-generation') || $('test-provider-generation').disabled) return;
    $('test-provider-generation').disabled = true;
    $('runtime-detail').textContent = 'يجري إرسال اختبار JSON ثابت لا يتضمن معلومات الحالة…';
    try {
      const result = await api('/api/provider/test', { method: 'POST', body: '{}' });
      providerGenerationTest = !!result.success;
      if (result.success) {
        $('runtime-mode').textContent = `${providerConfig?.local ? 'نموذج Ollama المحلي' : 'مزود خارجي'} — نجح اختبار التوليد الثابت.`;
        $('runtime-detail').textContent = `نجح توليد JSON الثابت من ${providerLabel(result.provider)} (${result.model}) خلال ${result.latency_ms} مللي ثانية.`;
        setStatus('نجح اختبار توليد ثابت؛ لم تُرسل بيانات حالة.', true);
      } else {
        $('runtime-mode').textContent = `القواعد المحلية — فشل توليد ${providerLabel(result.provider)}.`;
        $('runtime-detail').textContent = `فشل اختبار التوليد (${result.failure_code || 'استجابة غير مقبولة'}). لا يثبت إعداد المتغيرات نجاح النموذج.`;
        setStatus('فشل اختبار توليد النموذج؛ ستبقى المعالجة بالقواعد أو المصادر المحلية.', false);
      }
    } catch (error) {
      providerGenerationTest = false;
      $('runtime-detail').textContent = `تعذر اختبار التوليد: ${error.message}`;
      setStatus('تعذر اختبار توليد النموذج.', false);
    } finally { $('test-provider-generation').disabled = !(providerProbe?.service_reachable && providerProbe?.model_available); }
  }
  async function checkHealth() {
    try {
      const health = await api('/api/health');
      renderProviderConfig(health.provider);
    } catch {
      renderProviderConfig(null);
      $('runtime-detail').textContent = 'تعذر الوصول إلى واجهة التطبيق؛ لم يُفحص مزود النموذج.';
    }
  }
  function updateConsentVisibility() {
    $('model-consent-row').classList.toggle('hidden', !externalConfigured);
    $('model-consent').checked = false;
    const target = providerConfig?.local ? 'نموذج Ollama المحلي المضبوط' : 'مزود خارجي';
    $('model-consent-copy').textContent = kind === 'question'
      ? `أوافق على إرسال نص هذا السؤال الحالي ومقتطفات المصادر المختارة فقط إلى ${target} لهذا الرد. لن يُرسل سجل المقابلة.`
      : `أوافق على إرسال نص رسالتي الحالية فقط إلى ${target} لاستخراج حقائق منظمة. قد يتضمن النص بيانات شخصية؛ لن يُرسل سجل المحادثة أو إجابات سابقة.`;
  }

  $('start-case').addEventListener('click', createCase);
  $('new-case').addEventListener('click', createCase);
  $('chat-form').addEventListener('submit', sendMessage);
  $('pause-case').addEventListener('click', pauseOrResume);
  $('confirm-summary').addEventListener('click', confirmSummary);
  $('download-report').addEventListener('click', downloadReport);
  $('download-json').addEventListener('click', () => downloadReport('json'));
  $('print-report').addEventListener('click', printReport);
  $('check-provider').addEventListener('click', checkProvider);
  $('test-provider-generation').addEventListener('click', testProviderGeneration);
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
