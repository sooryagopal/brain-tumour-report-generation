/* ─────────────────────────────────────────────────────────────────────────────
   ProbabilityChart Component
   Horizontal bar chart showing class probability distribution
───────────────────────────────────────────────────────────────────────────── */

const CLASS_COLORS = {
  glioma:       '#6366f1',
  meningioma:   '#8b5cf6',
  pituitary:    '#06b6d4',
  no_tumor:     '#22c55e',
  astrocytoma:  '#f59e0b',
  ependymoma:   '#ec4899',
};

const CLASS_LABELS = {
  glioma:       'Glioma',
  meningioma:   'Meningioma',
  pituitary:    'Pituitary',
  no_tumor:     'No Tumour',
  astrocytoma:  'Astrocytoma',
  ependymoma:   'Ependymoma',
};

export default function ProbabilityChart({ probabilities, topClass }) {
  if (!probabilities) return null;

  // Sort by probability descending
  const sorted = Object.entries(probabilities)
    .sort(([, a], [, b]) => b - a);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      {sorted.map(([cls, prob]) => {
        const isTop = cls === topClass;
        const color = CLASS_COLORS[cls] || '#6366f1';
        const pct = (prob * 100).toFixed(1);

        return (
          <div key={cls}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '6px',
              }}
            >
              <span
                style={{
                  fontSize: '0.82rem',
                  fontWeight: isTop ? '700' : '500',
                  color: isTop ? 'var(--text-primary)' : 'var(--text-secondary)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                }}
              >
                {isTop && (
                  <span
                    style={{
                      width: 8,
                      height: 8,
                      background: color,
                      borderRadius: '50%',
                      display: 'inline-block',
                    }}
                  />
                )}
                {CLASS_LABELS[cls] || cls}
              </span>
              <span
                style={{
                  fontSize: '0.82rem',
                  fontWeight: isTop ? '700' : '500',
                  color: isTop ? color : 'var(--text-muted)',
                  fontVariantNumeric: 'tabular-nums',
                }}
              >
                {pct}%
              </span>
            </div>

            <div className="progress-bar-track">
              <div
                className="progress-bar-fill"
                style={{
                  width: `${pct}%`,
                  background: isTop
                    ? `linear-gradient(90deg, ${color}, ${color}cc)`
                    : `rgba(255,255,255,0.1)`,
                  boxShadow: isTop ? `0 0 10px ${color}66` : 'none',
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
