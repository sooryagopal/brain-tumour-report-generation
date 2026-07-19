/* ─────────────────────────────────────────────────────────────────────────────
   SeverityBadge Component
   Renders a colored badge for: Severe / Moderate / Mild / Normal
───────────────────────────────────────────────────────────────────────────── */

const CONFIG = {
  Severe:   { cls: 'badge-severe',   emoji: '🔴', label: 'Severe' },
  Moderate: { cls: 'badge-moderate', emoji: '🟡', label: 'Moderate' },
  Mild:     { cls: 'badge-mild',     emoji: '🟢', label: 'Mild' },
  Normal:   { cls: 'badge-normal',   emoji: '✅', label: 'Normal' },
};

export default function SeverityBadge({ severity, size = 'md' }) {
  const cfg = CONFIG[severity] || { cls: 'badge-indigo', emoji: '⚪', label: severity };

  const fontSizeMap = { sm: '0.7rem', md: '0.8rem', lg: '1rem' };
  const paddingMap  = { sm: '3px 10px', md: '4px 14px', lg: '6px 18px' };

  return (
    <span
      className={`badge ${cfg.cls}`}
      style={{ fontSize: fontSizeMap[size], padding: paddingMap[size] }}
    >
      {cfg.emoji} {cfg.label}
    </span>
  );
}
