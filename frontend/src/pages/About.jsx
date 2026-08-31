/* ─────────────────────────────────────────────────────────────────────────────
   About Page — Project info, architecture, team, methodology
───────────────────────────────────────────────────────────────────────────── */

import { GraduationCap, BookOpen, Users, Cpu, GitBranch, Target } from 'lucide-react';

const PIPELINE = [
  { step: '01', title: 'MRI Upload',         desc: 'JPG/PNG brain MRI scan uploaded via drag-and-drop or file browser.', icon: '📤' },
  { step: '02', title: 'Preprocessing',       desc: 'CLAHE contrast enhancement, ROI extraction, gamma correction, normalisation to 224×224.', icon: '⚙️' },
  { step: '03', title: 'CNN Ensemble',  desc: 'EfficientNet-B4 + DenseNet-121 + ResNet-50 ensemble fine-tuned on 7,000+ images across 6 tumour classes.', icon: '🧠' },
  { step: '04', title: 'Severity Grading',    desc: 'Confidence-grounded clinical rules map tumour type + confidence → Mild/Moderate/Severe.', icon: '📊' },
  { step: '05', title: 'LLM Report Gen',      desc: 'Ollama LLM generates structured Findings / Impression / Recommendation sections grounded by classifier output.', icon: '📋' },
  { step: '06', title: 'NLP Evaluation',      desc: 'BLEU-1, ROUGE-L, BERTScore metrics evaluate report quality against reference reports.', icon: '📈' },
];

const TECH = [
  { name: 'EfficientNet + DenseNet + ResNet', category: 'CNN Ensemble',         color: '#6366f1' },
  { name: 'FastAPI',          category: 'Backend API',          color: '#06b6d4' },
  { name: 'React + Vite',     category: 'Frontend',             color: '#f59e0b' },
  { name: 'Ollama LLM',       category: 'Report Generation',    color: '#8b5cf6' },
  { name: 'PyTorch',          category: 'Deep Learning',        color: '#ef4444' },
  { name: 'OpenCV + CLAHE',   category: 'Image Preprocessing',  color: '#22c55e' },
  { name: 'BERTScore',        category: 'NLP Evaluation',       color: '#ec4899' },
  { name: 'ROUGE-L / BLEU',   category: 'Text Metrics',         color: '#0d9488' },
];

const DATASETS = [
  { name: 'Kaggle — Masoud Nickparvar',  classes: '4 classes',   images: '7,023',  link: 'https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset' },
  { name: 'Kaggle — Fernando 17-Class',  classes: '17 classes',  images: '~4,000', link: 'https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-17-classes' },
  { name: 'Figshare Brain Tumor',        classes: '3 classes',   images: '3,064',  link: 'https://figshare.com/articles/dataset/brain_tumor_dataset/1512427' },
];

const NOVEL = [
  'Multi-class classification across 6 tumour types (unlike Paper 1: single class label)',
  'Confidence-grounded severity grading — first such integration in the literature',
  'Structured LLM report (Findings/Impression/Recommendation) with hard grounding constraint',
  '5.1% lower hallucination rate vs free-form generation (Park et al., 2026)',
  'Combined BLEU-1, ROUGE-L, BERTScore evaluation pipeline for radiology NLP',
];

export default function About() {
  return (
    <div style={{ padding: '48px 0 100px' }}>
      <div className="container">

        {/* Header */}
        <div style={{ marginBottom: '56px', maxWidth: 700 }}>
          <div className="badge badge-indigo" style={{ marginBottom: '16px', display: 'inline-flex' }}>
            <GraduationCap size={11} />
            Final Year Project — B.E. AI & DS 2027
          </div>
          <h1 style={{ fontSize: 'clamp(2rem, 5vw, 3rem)', marginBottom: '16px' }}>
            About{' '}
            <span className="gradient-text">NeuroScan AI</span>
          </h1>
          <p style={{ fontSize: '1rem', color: 'var(--text-secondary)', lineHeight: 1.8, maxWidth: 620 }}>
            <strong style={{ color: 'var(--text-primary)' }}>Multi-Class Brain Tumour MRI Report Generation with Severity Grading Using Fine-Tuned Vision-Language Models</strong>
            <br /><br />
            Project <strong style={{ color: 'var(--color-indigo)' }}>22ADP72</strong> — Kongu Engineering College | B.E. Artificial Intelligence & Data Science | Batch 2027
          </p>
        </div>

        {/* Novel Contribution */}
        <div className="glass-card" style={{ padding: '32px', marginBottom: '48px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
            <Target size={20} color="var(--color-indigo)" />
            <h2 style={{ fontSize: '1.2rem' }}>Novel Contribution</h2>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {NOVEL.map((point, i) => (
              <div key={i} style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                <div
                  style={{
                    width: 22, height: 22, borderRadius: '50%',
                    background: 'var(--grad-primary)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '0.68rem', fontWeight: 700, color: '#fff',
                    flexShrink: 0, marginTop: '1px',
                  }}
                >
                  {i + 1}
                </div>
                <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.6 }}>
                  {point}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Pipeline */}
        <div style={{ marginBottom: '56px' }}>
          <h2 style={{ fontSize: '1.6rem', marginBottom: '8px' }}>
            System <span className="gradient-text">Pipeline</span>
          </h2>
          <p style={{ color: 'var(--text-muted)', marginBottom: '28px', fontSize: '0.9rem' }}>
            End-to-end processing from raw MRI upload to structured radiology report.
          </p>
          <div style={{ position: 'relative' }}>
            {PIPELINE.map(({ step, title, desc, icon }, i) => (
              <div
                key={step}
                style={{
                  display: 'flex',
                  gap: '20px',
                  alignItems: 'flex-start',
                  marginBottom: i < PIPELINE.length - 1 ? '0' : '0',
                  position: 'relative',
                }}
              >
                {/* Connector line */}
                {i < PIPELINE.length - 1 && (
                  <div
                    style={{
                      position: 'absolute',
                      left: '19px',
                      top: '48px',
                      width: '2px',
                      height: '40px',
                      background: 'linear-gradient(180deg, var(--color-indigo), transparent)',
                    }}
                  />
                )}

                <div
                  style={{
                    width: 40, height: 40,
                    borderRadius: '12px',
                    background: 'var(--grad-primary)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '0.7rem', fontWeight: 800, color: '#fff',
                    flexShrink: 0, letterSpacing: '0.02em',
                  }}
                >
                  {step}
                </div>

                <div
                  className="glass-card"
                  style={{
                    flex: 1,
                    padding: '16px 20px',
                    marginBottom: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '14px',
                  }}
                >
                  <span style={{ fontSize: '20px' }}>{icon}</span>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)', marginBottom: '3px' }}>{title}</div>
                    <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>{desc}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Tech Stack */}
        <div style={{ marginBottom: '56px' }}>
          <h2 style={{ fontSize: '1.6rem', marginBottom: '8px' }}>
            Technology <span className="gradient-text">Stack</span>
          </h2>
          <p style={{ color: 'var(--text-muted)', marginBottom: '28px', fontSize: '0.9rem' }}>
            State-of-the-art tools chosen for medical AI and production-readiness.
          </p>
          <div className="grid-4">
            {TECH.map(({ name, category, color }) => (
              <div
                key={name}
                className="glass-card"
                style={{ padding: '20px', textAlign: 'center' }}
              >
                <div
                  style={{
                    width: 44, height: 44, borderRadius: '12px',
                    background: `${color}20`, border: `1px solid ${color}40`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    margin: '0 auto 12px',
                  }}
                >
                  <Cpu size={18} color={color} />
                </div>
                <div style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--text-primary)', marginBottom: '4px' }}>
                  {name}
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  {category}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Datasets */}
        <div style={{ marginBottom: '56px' }}>
          <h2 style={{ fontSize: '1.6rem', marginBottom: '8px' }}>
            <span className="gradient-text">Datasets</span> Used
          </h2>
          <p style={{ color: 'var(--text-muted)', marginBottom: '28px', fontSize: '0.9rem' }}>
            6-class dataset assembled from 3 public sources covering 7,000+ annotated MRI images.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {DATASETS.map(({ name, classes, images, link }) => (
              <a
                key={name}
                href={link}
                target="_blank"
                rel="noopener noreferrer"
                className="glass-card"
                style={{
                  padding: '18px 24px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  textDecoration: 'none',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  flexWrap: 'wrap',
                  gap: '12px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <BookOpen size={18} color="var(--color-indigo)" />
                  <span style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)' }}>{name}</span>
                </div>
                <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
                  <span className="badge badge-cyan">{classes}</span>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{images} images</span>
                  <span style={{ fontSize: '0.78rem', color: 'var(--color-indigo)' }}>↗ Kaggle</span>
                </div>
              </a>
            ))}
          </div>
        </div>

        {/* Project Info Footer */}
        <div
          className="glass-card"
          style={{
            padding: '32px',
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '24px',
            textAlign: 'center',
          }}
        >
          <InfoTile icon={<GraduationCap size={20} />} label="Institution" value="Kongu Engineering College" />
          <InfoTile icon={<Users size={20} />} label="Department" value="B.E. AI & Data Science" />
          <InfoTile icon={<GitBranch size={20} />} label="Project ID" value="22ADP72 — Batch 2027" />
        </div>

      </div>
    </div>
  );
}

function InfoTile({ icon, label, value }) {
  return (
    <div>
      <div style={{ color: 'var(--color-indigo)', marginBottom: '8px', display: 'flex', justifyContent: 'center' }}>
        {icon}
      </div>
      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '4px' }}>
        {label}
      </div>
      <div style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)' }}>
        {value}
      </div>
    </div>
  );
}
