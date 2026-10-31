/*
 * Shared prototype engine for the three Faith ID UI proposals.
 *
 * Holds the sample data and the kiosk / server state machines; each proposal page only draws
 * screens from Proto.state and calls Proto actions. No camera: the test panel simulates who is in
 * front of the kiosk. All names are sample data.
 */
(function () {
  const DEPTS = [
    { id: 'baby', name: '영아부', color: '#f2a7b5' },
    { id: 'kinder', name: '유치부', color: '#f6c453' },
    { id: 'junior', name: '유년부', color: '#7cc7a4' },
    { id: 'elem', name: '초등부', color: '#6fa8e8' },
    { id: 'youth', name: '중고등부', color: '#a98be0' },
    { id: 'adult', name: '청년부', color: '#9aa3b5' },
  ];
  const GRADES = {
    baby: ['0세', '1세', '2세', '3세'],
    kinder: ['4세', '5세', '6세', '7세'],
    junior: ['1학년', '2학년', '3학년'],
    elem: ['4학년', '5학년', '6학년'],
    youth: ['중1', '중2', '중3', '고1', '고2', '고3'],
    adult: ['1년차', '2년차', '3년차', '리더'],
  };
  const SEED = [
    ['김하은', 'junior', '2학년'], ['이준호', 'elem', '5학년'], ['박서윤', 'kinder', '6세'],
    ['최도윤', 'junior', '1학년'], ['정민서', 'elem', '4학년'], ['강지후', 'kinder', '7세'],
    ['조예린', 'junior', '3학년'], ['윤시우', 'elem', '6학년'], ['장하린', 'baby', '3세'],
    ['임서준', 'youth', '중2'], ['한지아', 'junior', '2학년'], ['오태양', 'elem', '5학년'],
    ['신유나', 'kinder', '5세'], ['권도현', 'youth', '고1'], ['송다온', 'junior', '1학년'],
    ['김하윤', 'junior', '2학년'], ['문채원', 'elem', '4학년'], ['배주원', 'kinder', '6세'],
  ];
  let nextId = 1;
  const people = SEED.map(([name, dept, grade]) => ({ id: 'p' + nextId++, name, dept, grade, samples: 12 + (nextId * 7) % 30 }));
  // 김하은 and 김하윤 are the look-alike pair used by the "닮은 얼굴" scenario.
  const twinA = people.find(p => p.name === '김하은');
  const twinB = people.find(p => p.name === '김하윤');

  const now = new Date();
  const hhmm = d => String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
  const minutesAgo = m => new Date(now.getTime() - m * 60000);
  // Today's attendance so far (keyed by person id; guests by name).
  const attendance = [];
  [[2, 24], [5, 21], [4, 19], [7, 17], [9, 15], [3, 12], [12, 9], [6, 6], [10, 4]].forEach(([idx, ago], i) => {
    const p = people[idx];
    attendance.push({ key: p.id, person: p, first: minutesAgo(ago), last: minutesAgo(ago), count: 1, kiosk: i % 2 ? '유년부 입구' : '본당 로비' });
  });

  const kiosks = [
    { id: 'client-7a1c', name: '본당 로비', status: 'online', last: '방금' },
    { id: 'client-e8c5', name: '유년부 입구', status: 'online', last: '방금' },
    { id: 'client-3f90', name: '새 태블릿 (SM-T975N)', status: 'pending', last: '방금' },
  ];

  const DIRECTIONS = [
    { id: 'left', label: '왼쪽', arrow: '←', say: '왼쪽을 봐요' },
    { id: 'right', label: '오른쪽', arrow: '→', say: '오른쪽을 봐요' },
    { id: 'up', label: '위', arrow: '↑', say: '위를 봐요' },
    { id: 'down', label: '아래', arrow: '↓', say: '아래를 봐요' },
  ];
  const PER_DIRECTION = 3;

  const state = {
    view: 'kiosk',               // kiosk | server
    // kiosk
    screen: 'idle',              // idle | detecting | recognized | candidates | unknown | register | regName | saved | confirmSelf | labelOnly | printing | done
    face: null,                  // who is in front: { kind, person? }
    person: null,                // recognized or chosen person
    candidates: [],
    guest: null,                 // label-only content
    reg: { dir: 0, counts: [0, 0, 0, 0], match: null },
    form: { name: '', dept: '', grade: '' },
    countdown: 0,
    connection: 'online',        // online | offline | pending
    pendingSend: 0,
    voice: false,
    audience: 'kids',            // kids | adults: the same screens, a different tone
    lastCheckin: null,
    // server
    tab: 'home',
    search: '',
    deptFilter: '',
    toast: '',
    editing: null,
  };

  const listeners = [];
  let timers = [];
  function emit() { listeners.forEach(fn => { try { fn(state); } catch (e) { console.error(e); } }); }
  function clearTimers() { timers.forEach(clearTimeout); timers.forEach(clearInterval); timers = []; }
  function later(ms, fn) { const t = setTimeout(fn, ms); timers.push(t); return t; }

  function say(text) {
    if (!state.voice || !('speechSynthesis' in window)) return;
    try {
      speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = 'ko-KR'; u.rate = 1.0; u.pitch = 1.15;
      speechSynthesis.speak(u);
    } catch (e) { /* speech is optional */ }
  }
  function callName(p) { // kids: 하은아 / 준호야, adults: 김하은 님
    if (state.audience === 'adults') return p.name + ' 님';
    const given = p.name.slice(1);
    const last = given.charCodeAt(given.length - 1) - 0xac00;
    const hasFinal = last >= 0 && last < 11172 && last % 28 !== 0;
    return given + (hasFinal ? '아' : '야');
  }

  // Wording for both audiences; pages may override with their own lines.
  const COPY = {
    idle: { kids: '얼굴을 동그라미 안에 보여 줘요', adults: '화면을 바라보시면 출석이 확인됩니다' },
    hello: { kids: p => '반가워, ' + callName(p) + '!', adults: p => callName(p) + ', 환영합니다' },
    printCta: { kids: '이름표 받기', adults: '이름표 출력' },
    unknown: { kids: '처음 왔구나! 얼굴을 등록할까요?', adults: '등록되지 않은 얼굴입니다' },
    register: { kids: '얼굴 등록하기', adults: '얼굴 등록' },
    labelOnly: { kids: '이름표만 받을래요', adults: '등록 없이 이름표만' },
    regTurn: { kids: d => d.label + '을 봐요', adults: d => '고개를 ' + d.label + '으로 천천히 돌려 주세요' },
    regName: { kids: '거의 다 됐어요! 선생님이 이름을 적어 주세요', adults: '이름과 소속을 입력해 주세요' },
    saved: { kids: '등록 완료!', adults: '등록되었습니다' },
    twins: { kids: '누구인지 알려 줄래요? 내 이름을 눌러요', adults: '본인의 이름을 선택해 주세요' },
    confirm: { kids: p => callName(p) + ' 맞아요?', adults: p => p.name + ' 님이 맞으신가요?' },
    confirmYes: { kids: '네, 맞아요', adults: '네, 본인입니다' },
    confirmNo: { kids: '아니에요, 처음이에요', adults: '아니요, 처음입니다' },
    done: { kids: p => p ? '잘 왔어, ' + callName(p) + '! 이름표를 가져가요' : '이름표를 가져가요', adults: p => p ? '출석되었습니다. 이름표를 가져가세요' : '이름표를 가져가세요' },
    printing: { kids: '이름표가 나오고 있어요', adults: '이름표를 출력하고 있습니다' },
    pending: { kids: '선생님, 이 키오스크를 서버에서 허락해 주세요', adults: '이 키오스크는 서버 승인을 기다리고 있습니다' },
    offline: { kids: '출석은 저장해 두었다가 보낼게요', adults: '서버와 연결이 끊겨 출석을 보관 중입니다' },
    home: { kids: '처음으로', adults: '처음으로' },
  };
  function t(key, arg) {
    const v = COPY[key][state.audience];
    return typeof v === 'function' ? v(arg) : v;
  }

  function go(screen, extra) {
    Object.assign(state, { screen }, extra || {});
    emit();
  }

  function recordCheckin(entry) {
    const key = entry.person ? entry.person.id : 'guest:' + entry.name;
    const t = new Date();
    const row = attendance.find(r => r.key === key);
    if (row) { row.last = t; row.count += 1; }
    else attendance.push({ key, person: entry.person || null, guest: entry.person ? null : { name: entry.name, dept: entry.dept }, first: t, last: t, count: 1, kiosk: '본당 로비' });
    state.lastCheckin = { key, at: t };
    if (state.connection !== 'online') state.pendingSend += 1;
  }

  const Proto = {
    DEPTS, GRADES, DIRECTIONS, PER_DIRECTION, people, attendance, kiosks, state, hhmm, callName,
    dept: id => DEPTS.find(d => d.id === id) || { id, name: id, color: '#9aa3b5' },
    subscribe(fn) { listeners.push(fn); fn(state); },
    refresh: () => emit(),
    regDoneDelay: 0,             // pages may hold the finished registration screen longer, e.g. for a celebration
    regStepMs: 0,                // pause between samples of one direction (default 520 ms)
    regTurnMs: 0,                // pause after turning to a new direction (default 1100 ms)
    regStepSay: null,            // (direction, sampleIndex) => spoken line for each sample, if a page wants one
    t,
    setView(v) { state.view = v; emit(); },
    setAudience(a) { state.audience = a; emit(); },
    setVoice(on) { state.voice = on; if (!on && 'speechSynthesis' in window) speechSynthesis.cancel(); emit(); },

    /* ---------- test panel: who stands in front of the kiosk ---------- */
    simulate(kind) {
      clearTimers();
      state.form = { name: '', dept: '', grade: '' };
      if (kind === 'leave') { state.face = null; go('idle', { person: null, candidates: [] }); return; }
      if (state.connection === 'pending') { state.face = { kind }; emit(); return; }
      state.face = { kind };
      go('detecting', { person: null, candidates: [] });
      later(700, () => {
        if (kind === 'known') {
          const pool = people.filter(p => p !== twinA && p !== twinB);
          const p = pool[Math.floor(Math.random() * pool.length)];
          go('recognized', { person: p });
          say(t('hello', p) + ' ' + t('printCta') + ' 버튼을 눌러 주세요');
        } else if (kind === 'twins') {
          go('candidates', { candidates: [twinA, twinB] });
          say(t('twins'));
        } else if (kind === 'new' || kind === 'returning') {
          go('unknown');
          say(t('unknown'));
        }
      });
    },
    setConnection(c) { state.connection = c; if (c === 'online') state.pendingSend = 0; emit(); },

    /* ---------- kiosk actions ---------- */
    home() { clearTimers(); state.face = null; go('idle', { person: null, candidates: [], guest: null }); },
    pick(id) { const p = people.find(x => x.id === id); go('recognized', { person: p, candidates: [] }); },
    print() {
      const p = state.person;
      clearTimers();
      if (p) recordCheckin({ person: p });
      else if (state.guest) recordCheckin({ name: state.guest.name, dept: state.guest.dept });
      go('printing');
      later(1600, () => {
        go('done', { countdown: 4 });
        say(t('done', p));
        const tick = setInterval(() => {
          state.countdown -= 1;
          if (state.countdown <= 0) { clearInterval(tick); Proto.home(); } else emit();
        }, 1000);
        timers.push(tick);
      });
    },
    startRegister() {
      clearTimers();
      state.reg = { dir: 0, counts: [0, 0, 0, 0], match: state.face && state.face.kind === 'returning' ? people[0] : null };
      go('register');
      say(Proto.regStepSay ? Proto.regStepSay(DIRECTIONS[0], 0) : t('regTurn', DIRECTIONS[0]));
      const step = () => {
        const r = state.reg;
        if (r.dir >= DIRECTIONS.length) return;
        r.counts[r.dir] += 1;
        if (r.counts[r.dir] >= PER_DIRECTION) {
          r.dir += 1;
          if (r.dir < DIRECTIONS.length) say(Proto.regStepSay ? Proto.regStepSay(DIRECTIONS[r.dir], 0) : t('regTurn', DIRECTIONS[r.dir]));
        } else if (Proto.regStepSay) say(Proto.regStepSay(DIRECTIONS[r.dir], r.counts[r.dir]));
        if (r.dir >= DIRECTIONS.length) {
          later(Proto.regDoneDelay || 500, () => {
            if (r.match) { go('confirmSelf', { person: r.match }); say(t('confirm', r.match)); }
            else { go('regName'); say(t('regName')); }
          });
          emit();
          return;
        }
        emit();
        later(r.counts[r.dir] === 0 ? (Proto.regTurnMs || 1100) : (Proto.regStepMs || 520), step);
      };
      later(900, step);
    },
    setForm(patch) { Object.assign(state.form, patch); emit(); },
    saveRegistration() {
      const f = state.form;
      if (!f.name.trim()) return false;
      const p = { id: 'p' + nextId++, name: f.name.trim(), dept: f.dept || 'junior', grade: f.grade || '', samples: 12, isNew: true };
      people.push(p);
      go('saved', { person: p });
      say(t('saved'));
      later(1400, () => go('recognized', { person: p }));
      return true;
    },
    confirmSelf(yes) {
      if (yes) go('recognized', { person: state.person });
      else { go('regName', { person: null }); }
    },
    startLabelOnly() { clearTimers(); state.form = { name: '', dept: '', grade: '' }; go('labelOnly', { person: null }); },
    printLabelOnly() {
      const f = state.form;
      if (!f.name.trim()) return false;
      state.guest = { name: f.name.trim(), dept: f.dept ? Proto.dept(f.dept).name : (f.deptText || ''), grade: f.grade };
      state.person = null;
      Proto.print();
      return true;
    },

    /* ---------- server actions ---------- */
    setTab(t) { state.tab = t; state.editing = null; emit(); },
    setSearch(q) { state.search = q; emit(); },
    setDeptFilter(d) { state.deptFilter = state.deptFilter === d ? '' : d; emit(); },
    approve(id) { const k = kiosks.find(x => x.id === id); if (k) { k.status = 'online'; toast(k.name + ' 연결을 허락했어요'); } },
    remove(id) { const i = kiosks.findIndex(x => x.id === id); if (i >= 0) { const [k] = kiosks.splice(i, 1); toast(k.name + ' 연결을 해제했어요'); } },
    edit(id) { state.editing = id; emit(); },
    savePerson(id, patch) { const p = people.find(x => x.id === id); Object.assign(p, patch); state.editing = null; toast(p.name + ' 정보를 저장했어요. 모든 키오스크에 반영됩니다.'); },
    deletePerson(id) { const i = people.findIndex(x => x.id === id); if (i >= 0) { const [p] = people.splice(i, 1); state.editing = null; toast(p.name + '을(를) 명단에서 지웠어요'); } },
    toast,

    visiblePeople() {
      const q = state.search.trim();
      return people.filter(p => (!state.deptFilter || p.dept === state.deptFilter) &&
        (!q || (p.name + Proto.dept(p.dept).name + p.grade).includes(q)));
    },
    attendanceByDept() {
      return DEPTS.map(d => {
        const roster = people.filter(p => p.dept === d.id);
        const present = attendance.filter(a => a.person && a.person.dept === d.id);
        return { dept: d, roster: roster.length, present: present.length, rows: present };
      }).filter(g => g.roster > 0);
    },
    guests() { return attendance.filter(a => !a.person); },

    /* ---------- layout helper: scale a fixed 1280×800 screen into its frame ---------- */
    fit(frame, screen, width = 1280, height = 800) {
      const apply = () => {
        const s = frame.clientWidth / width;
        screen.style.transform = 'scale(' + s + ')';
        frame.style.height = (height * s) + 'px';
      };
      apply();
      if ('ResizeObserver' in window) new ResizeObserver(apply).observe(frame);
      else window.addEventListener('resize', apply);
    },
  };

  function toast(msg) {
    state.toast = msg; emit();
    const t = setTimeout(() => { if (state.toast === msg) { state.toast = ''; emit(); } }, 2600);
    timers.push(t);
  }


  /* ---------- page shell: test panel, device frame, render loop ---------- */
  // opts: { root, kiosk: state => html, server: state => html, views?: [{ id, label, render: state => html }] }
  // Extra views sit between the kiosk and the server app in the view switch and use the kiosk scenarios.
  Proto.mount = function (opts) {
    const root = opts.root;
    const extra = opts.views || [];
    root.innerHTML = `
      <div class="lab" role="region" aria-label="테스트 도구">
        <div class="lab-row">
          <div class="seg" role="tablist" aria-label="화면">
            <button type="button" data-lab="view" data-v="kiosk" role="tab">키오스크</button>
            ${extra.map(v => `<button type="button" data-lab="view" data-v="${v.id}" role="tab">${v.label}</button>`).join('')}
            <button type="button" data-lab="view" data-v="server" role="tab">서버 앱</button>
          </div>
          <div class="seg" aria-label="대상">
            <button type="button" data-lab="aud" data-v="kids">어린이</button>
            <button type="button" data-lab="aud" data-v="adults">어른</button>
          </div>
          <label class="lab-check"><input type="checkbox" id="lab-voice" data-lab="voice"> 음성 안내</label>
        </div>
        <div class="lab-row kiosk-only">
          <span class="lab-label">카메라 앞에</span>
          <button type="button" data-lab="sim" data-v="known">등록된 아이</button>
          <button type="button" data-lab="sim" data-v="new">처음 온 아이</button>
          <button type="button" data-lab="sim" data-v="twins">닮은 형제</button>
          <button type="button" data-lab="sim" data-v="returning">등록했는데 다시 등록</button>
          <button type="button" data-lab="sim" data-v="leave">아무도 없음</button>
        </div>
        <div class="lab-row kiosk-only">
          <span class="lab-label">서버 연결</span>
          <button type="button" data-lab="conn" data-v="online">연결됨</button>
          <button type="button" data-lab="conn" data-v="offline">끊김</button>
          <button type="button" data-lab="conn" data-v="pending">승인 대기</button>
        </div>
      </div>
      <div class="device"><div class="device-screen" id="screen"></div></div>`;
    const frame = root.querySelector('.device');
    const screen = root.querySelector('#screen');
    Proto.fit(frame, screen);

    root.addEventListener('click', e => {
      const lab = e.target.closest('[data-lab]');
      if (lab && lab.dataset.lab !== 'voice') {
        const v = lab.dataset.v;
        if (lab.dataset.lab === 'view') Proto.setView(v);
        if (lab.dataset.lab === 'aud') Proto.setAudience(v);
        if (lab.dataset.lab === 'sim') { if (state.view === 'server') Proto.setView('kiosk'); Proto.simulate(v); }
        if (lab.dataset.lab === 'conn') Proto.setConnection(v);
        return;
      }
      const el = e.target.closest('[data-act]');
      if (!el || !screen.contains(el)) return;
      const id = el.dataset.id;
      const acts = {
        home: () => Proto.home(), print: () => Proto.print(), pick: () => Proto.pick(id),
        register: () => Proto.startRegister(), labelonly: () => Proto.startLabelOnly(),
        'confirm-yes': () => Proto.confirmSelf(true), 'confirm-no': () => Proto.confirmSelf(false),
        save: () => { if (!Proto.saveRegistration()) Proto.toast('이름을 입력해 주세요'); },
        printlabel: () => { if (!Proto.printLabelOnly()) Proto.toast('이름을 입력해 주세요'); },
        dept: () => Proto.setForm({ dept: id, grade: '' }), grade: () => Proto.setForm({ grade: id }),
        tab: () => Proto.setTab(id), approve: () => Proto.approve(id), remove: () => Proto.remove(id),
        filter: () => Proto.setDeptFilter(id), edit: () => Proto.edit(id), 'cancel-edit': () => Proto.edit(null),
        saveperson: () => {
          const form = el.closest('[data-editform]');
          const val = n => (form.querySelector('[data-edit="' + n + '"]') || {}).value || '';
          Proto.savePerson(id, { name: val('name') || undefined, dept: val('dept'), grade: val('grade') });
        },
        deleteperson: () => Proto.deletePerson(id),
        view: () => Proto.setView(id),
      };
      if (acts[el.dataset.act]) acts[el.dataset.act]();
    });
    root.addEventListener('change', e => {
      if (e.target.dataset.lab === 'voice') Proto.setVoice(e.target.checked);
    });
    // Typing updates state without redrawing, so the field keeps focus.
    screen.addEventListener('input', e => {
      const f = e.target.dataset.field;
      if (f === 'search') { state.search = e.target.value; render(true); return; }
      if (f) state.form[f] = e.target.value;
    });

    function render(keepFocus) {
      const active = document.activeElement;
      const field = active && screen.contains(active) ? active.dataset.field : null;
      const caret = field ? active.selectionStart : null;
      screen.dataset.view = state.view;
      screen.dataset.aud = state.audience;
      const view = extra.find(v => v.id === state.view);
      screen.innerHTML = view ? view.render(state) : state.view === 'kiosk' ? opts.kiosk(state) : opts.server(state);
      root.querySelectorAll('[data-lab="view"]').forEach(b => b.setAttribute('aria-selected', String(b.dataset.v === state.view)));
      root.querySelectorAll('[data-lab="aud"]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.v === state.audience)));
      root.querySelectorAll('[data-lab="conn"]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.v === state.connection)));
      root.querySelectorAll('.kiosk-only').forEach(r => { r.hidden = state.view === 'server'; });
      if (field) {
        const again = screen.querySelector('[data-field="' + field + '"]');
        if (again) { again.focus(); try { again.setSelectionRange(caret, caret); } catch (e) {} }
      }
    }
    listeners.push(() => render());
    render();
  };

  window.Proto = Proto;
})();
