const byId = (id) => document.getElementById(id);

function renderIntents(intents) {
  const container = byId('intent-list');
  container.innerHTML = intents.length ? intents.map((item) => `
    <div class="intent-row"><div class="intent-label"><span>${item.intent}</span><strong>${item.total} (${item.percentage}%)</strong></div>
    <div class="bar-bg"><div class="bar" style="width:${item.percentage}%"></div></div></div>`).join('') : '<p class="empty">Belum ada data percakapan.</p>';
}

function renderUnknown(items) {
  const list = byId('unknown-list');
  list.innerHTML = items.length ? items.map((item) => `<tr><td>${escapeHtml(item.user_message)}</td><td>${item.similarity_score == null ? '-' : item.similarity_score}</td><td>${item.created_at || '-'}</td></tr>`).join('') : '<tr><td colspan="3" class="empty">Belum ada pertanyaan tidak dikenali.</td></tr>';
}

function escapeHtml(value) { const div = document.createElement('div'); div.textContent = value; return div.innerHTML; }

async function loadStatistics() {
  byId('error').textContent = '';
  try {
    const response = await fetch('/api/statistics');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Gagal memuat statistik.');
    byId('total').textContent = data.summary.total;
    byId('recognized').textContent = data.summary.recognized;
    byId('unknown').textContent = data.summary.unknown;
    byId('popular').textContent = data.summary.most_common_intent || '-';
    renderIntents(data.intents); renderUnknown(data.unknown_questions);
  } catch (error) { byId('error').textContent = error.message; }
}

byId('refresh').addEventListener('click', loadStatistics);
loadStatistics();
