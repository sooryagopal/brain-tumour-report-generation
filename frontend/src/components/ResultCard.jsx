/* ─────────────────────────────────────────────────────────────────────────────
   ResultCard Component
   Displays classification result: class name, confidence, probabilities
───────────────────────────────────────────────────────────────────────────── */

import SeverityBadge from './SeverityBadge';
import ProbabilityChart from './ProbabilityChart';
import { TrendingUp, Target, AlertTriangle } from 'lucide-react';

const CLASS_LABELS = {
  glioma:       'Glioma',
  meningioma:   'Meningioma',
  pituitary:    'Pituitary Tumour',
  no_tumor:     'No Tumour Detected',
  astrocytoma:  'Astrocytoma',
  ependymoma:   'Ependymoma',
};

const CLASS_ICONS = {
  glioma:       '🧬',
  meningioma:   '🔬',
  pituitary:    '🫀',
  no_tumor:     '✅',
  astrocytoma:  '⚗️',
  ependymoma:   '🧪',
};

const SEVERITY_DESC_SHORT = {
  Severe:   'Immediate specialist referral required',
  Moderate: 'Further diagnostic workup recommended',
  Mild:     'Monitoring and follow-up advised',
  Normal:   'No tumour detected — routine follow-up',
};

export default function ResultCard({ result, demoMode }) {
  if (!result) return null;

  const {
    tumor_class: tumorClass,
    confidence,
    all_probabilities: probs,
    severity,
    severity_description: sevDesc,
    tumour_info: info,
  } = result;

  const classLabel = CLASS_LABELS[tumorClass] || tumorClass;
  const classIcon  = CLASS_ICONS[tumorClass]  || '🧠';
  const confPct    = (confidence * 100).toFixed(1);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

      {/* Demo Warning */}
      {demoMode && (
        <div className="demo-banner">
          <AlertTriangle size={15} />
          <span>
            <strong>Demo Mode</strong> — Model weights not found. Showing synthetic predictions.
            Add <code style={{ fontFamily: 'monospace', fontSize: '0.78rem' }}>models/efficientnet_b4.pth</code> to enable real inference.
          </span>
        </div>
      )}

      {/* Main Classification Result */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div
              style={{
                width: 52, height: 52,
                borderRadius: '14px',
                background: 'var(--grad-primary)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '24px',
              }}
            >
              {classIcon}
            </div>
            <div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '4px' }}>
                Classification Result
              </div>
              <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                {classLabel}
              </h2>
            </div>
          </div>
          <SeverityBadge severity={severity} size="md" />
        </div>

        {/* Metrics Row */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginBottom: '20px' }}>
          <MetricTile
            icon={<Target size={16} />}
            label="Confidence"
            value={`${confPct}%`}
            color="var(--color-indigo)"
          />
          <MetricTile
            icon={<TrendingUp size={16} />}
            label="Severity Grade"
            value={severity}
            color={
              severity === 'Severe'   ? 'var(--sev-severe)' :
              severity === 'Moderate' ? 'var(--sev-moderate)' :
              severity === 'Mild'     ? 'var(--sev-mild)' :
              'var(--sev-normal)'
            }
          />
          <MetricTile
            icon={<span style={{ fontSize: '16px' }}>🧠</span>}
            label="WHO Grade"
            value={info?.who_grade || 'N/A'}
            color="var(--color-cyan)"
          />
        </div>

        {/* Clinical note */}
        <div
          style={{
            background: 'rgba(255,255,255,0.03)',
            borderRadius: '10px',
            padding: '12px 16px',
            fontSize: '0.82rem',
            color: 'var(--text-muted)',
            borderLeft: '3px solid var(--color-indigo)',
          }}
        >
          {SEVERITY_DESC_SHORT[severity] || sevDesc}
        </div>
      </div>

      {/* Probability Chart */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <TrendingUp size={16} />
          Class Probability Distribution
        </h3>
        <ProbabilityChart probabilities={probs} topClass={tumorClass} />
      </div>

      {/* Tumour Info */}
      {info && info.description && (
        <div className="glass-card" style={{ padding: '20px 24px' }}>
          <h3 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '10px' }}>
            Clinical Reference
          </h3>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: '1.7' }}>
            {info.description}
          </p>
          {info.common_location !== 'N/A' && (
            <div style={{ marginTop: '10px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              📍 Common location: <strong style={{ color: 'var(--text-secondary)' }}>{info.common_location}</strong>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function MetricTile({ icon, label, value, color }) {
  return (
    <div
      style={{
        background: 'rgba(255,255,255,0.04)',
        borderRadius: '10px',
        padding: '14px',
        textAlign: 'center',
      }}
    >
      <div style={{ color: 'var(--text-muted)', marginBottom: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px', fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
        {icon} {label}
      </div>
      <div style={{ fontSize: '1.25rem', fontWeight: 700, color, fontFamily: 'Space Grotesk, sans-serif' }}>
        {value}
      </div>
    </div>
  );
}
