/* Bounded polling of a REAL job status. Stops on done/failed/cancelled/timeout or after the server-provided limit;
   never reports completion before /api/v2/jobs/<id> says so. */
(() => {
  'use strict';
  const box = document.getElementById('job'), out = document.getElementById('job-status');
  const id = box.dataset.jobId, limitMs = Number(box.dataset.timeout) * 1000, t0 = Date.now();
  const csrf = (document.querySelector('meta[name="csrf-token"]') || {}).content || '';
  let delay = 1500, stopped = false;
  const stop = msg => { stopped = true; out.textContent = msg; };
  document.getElementById('job-cancel').addEventListener('click', async e => {
    e.preventDefault();
    try { const r = await fetch(`/api/v2/jobs/${id}/cancel`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrf }, body: '{}' }); const j = await r.json();
      out.textContent = j.cancel_requested ? 'Cancel requested; a model fit already in progress finishes first.' : 'Nothing to cancel.'; } catch { out.textContent = 'Could not reach the server to cancel.'; }
  });
  async function tick() {
    if (stopped) return;
    if (Date.now() - t0 > limitMs) return stop('Gave up waiting: this job exceeded the time limit. Try again, or use the Forecast Lab.');
    try {
      const r = await fetch(`/api/v2/jobs/${id}`); if (!r.ok) return stop(`Status unavailable (${r.status}). The job may have expired.`);
      const j = await r.json();
      if (j.status === 'done' || j.status === 'failed' || j.status === 'cancelled') { stop('Finished. Loading…'); location.href = `/predict/claim/${id}`; return; }
      if (j.status === 'timeout') return stop('The job is taking longer than the allowed time and was reported as timed out.');
      out.textContent = `${j.status}: ${j.stage} (${Math.round((Date.now() - t0) / 1000)}s)`;
    } catch { out.textContent = 'Server unreachable; retrying…'; }
    delay = Math.min(delay * 1.3, 5000); setTimeout(tick, delay);
  }
  tick();
})();
