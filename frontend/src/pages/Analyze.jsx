/* ─────────────────────────────────────────────────────────────────────────────
   Analyze Page — Main MRI analysis workflow
   1. Upload MRI image (drag & drop)
   2. Patient info form
   3. Submit → backend API
   4. Show classification result + report
   5. Optional: Evaluate generated report vs reference
───────────────────────────────────────────────────────────────────────────── */

import { useState, useRef, useCallback } from 'react';
import axios from 'axios';
import {
  Upload, Scan, X, RefreshCw, ChevronDown, ChevronUp,
  BarChart2, AlertCircle, CheckCircle
} from 'lucide-react';
import ResultCard from '../components/ResultCard';
import ReportCard from '../components/ReportCard';

const API = '/api';

export default function Analyze() {
  // ── State ────────────────────────────────────────────────────────
  const [file, setFile]               = useState(null);
  const [preview, setPreview]         = useState(null);
  const [dragging, setDragging]       = useState(false);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);
  const [result, setResult]           = useState(null);

  // Patient form
  const [patientAge, setPatientAge]   = useState('');
  const [patientSex, setPatientSex]   = useState('');

  // Evaluation panel
  const [evalOpen, setEvalOpen]       = useState(false);
  const [refReport, setRefReport]     = useState('');
  const [evalResult, setEvalResult]   = useState(null);
  const [evalLoading, setEvalLoading] = useState(false);

  const inputRef = useRef(null);

  // ── Handlers ─────────────────────────────────────────────────────

  const handleFile = useCallback((f) => {
    if (!f) return;
    if (!['image/jpeg', 'image/jpg', 'image/png', 'image/webp'].includes(f.type)) {
      setError('Please upload a JPG, PNG, or WEBP image.');
      return;
    }
    setFile(f);
    setError(null);
    setResult(null);
    const url = URL.createObjectURL(f);
    setPreview(url);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    handleFile(f);
  }, [handleFile]);

  const onDragOver = (e) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);

  const clearFile = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setEvalResult(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);
    if (patientAge) formData.append('patient_age', patientAge);
    if (patientSex && patientSex !== 'Not specified') formData.append('patient_sex', patientSex);

    try {
      const { data } = await axios.post(`${API}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 90000,
      });
      setResult(data);
    } catch (err) {
      const msg =
        err?.response?.data?.detail ||
        err?.message ||
        'Analysis failed. Ensure the backend is running on port 8000.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleEvaluate = async () => {
    if (!result || !refReport.trim()) return;
    setEvalLoading(true);
    setEvalResult(null);

    const formData = new FormData();
    formData.append('generated', result.report?.full_report || '');
    formData.append('reference', refReport);

    try {
      const { data } = await axios.post(`${API}/evaluate`, formData);
      setEvalResult(data);
    } catch {
      setEvalResult({ error: 'Evaluation failed. Install nltk, rouge-score, bert-score.' });
    } finally {
      setEvalLoading(false);
    }
  };

  // ── Render ───────────────────────────────────────────────────────
  return (
    <div style={{ padding: '48px 0 100px' }}>
      <div className="container">

        {/* Page Header */}
        <div style={{ marginBottom: '40px' }}>
          <h1 style={{ fontSize: 'clamp(1.8rem, 4vw, 2.5rem)', marginBottom: '12px' }}>
            <span className="gradient-text">MRI Analysis</span> Workspace
          </h1>
          <p style={{ color: 'var(--text-secondary)', maxWidth: 520 }}>
            Upload a brain MRI image to classify the tumour type, grade severity, and generate a structured radiology report.
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '420px 1fr', gap: '28px', alignItems: 'start' }}>

          {/* ── Left Column: Upload + Patient Info ────────── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

            {/* Upload Zone */}
            <div className="glass-card" style={{ padding: '24px' }}>
              <h3 style={{ fontSize: '0.95rem', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)' }}>
                <Upload size={16} color="var(--color-indigo)" />
                Upload MRI Scan
              </h3>

              {preview ? (
                <div style={{ position: 'relative' }}>
                  <img
                    src={preview}
                    alt="MRI Preview"
                    style={{
                      width: '100%',
                      borderRadius: '12px',
                      maxHeight: '320px',
                      objectFit: 'contain',
                      background: 'rgba(0,0,0,0.4)',
                    }}
                  />
                  <button
                    onClick={clearFile}
                    style={{
                      position: 'absolute',
                      top: '8px',
                      right: '8px',
                      background: 'rgba(0,0,0,0.7)',
                      border: '1px solid rgba(255,255,255,0.2)',
                      borderRadius: '50%',
                      width: '30px',
                      height: '30px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: 'pointer',
                      color: '#fff',
                    }}
                  >
                    <X size={14} />
                  </button>
                  <div style={{ marginTop: '10px', fontSize: '0.78rem', color: 'var(--text-muted)', textAlign: 'center' }}>
                    {file.name} · {(file.size / 1024).toFixed(0)} KB
                  </div>
                </div>
              ) : (
                <div
                  className={`upload-zone${dragging ? ' drag-over' : ''}`}
                  onDrop={onDrop}
                  onDragOver={onDragOver}
                  onDragLeave={onDragLeave}
                  onClick={() => inputRef.current?.click()}
                >
                  <input
                    ref={inputRef}
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    style={{ display: 'none' }}
                    onChange={(e) => handleFile(e.target.files[0])}
                    id="mri-upload-input"
                  />
                  <div style={{ fontSize: '2.5rem', marginBottom: '12px' }}>🧠</div>
                  <div style={{ fontWeight: 600, fontSize: '0.95rem', color: 'var(--text-primary)', marginBottom: '6px' }}>
                    Drop MRI image here
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '14px' }}>
                    or click to browse · JPG, PNG, WEBP
                  </div>
                  <div className="badge badge-indigo" style={{ display: 'inline-flex' }}>
                    <Upload size={11} /> Select File
                  </div>
                </div>
              )}
            </div>

            {/* Patient Info */}
            <div className="glass-card" style={{ padding: '24px' }}>
              <h3 style={{ fontSize: '0.9rem', marginBottom: '16px', color: 'var(--text-secondary)' }}>
                Patient Information <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>(optional)</span>
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div>
                  <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>Age</label>
                  <input
                    id="patient-age-input"
                    type="number"
                    min={1}
                    max={120}
                    className="input-field"
                    placeholder="e.g. 45"
                    value={patientAge}
                    onChange={(e) => setPatientAge(e.target.value)}
                  />
                </div>
                <div>
                  <label style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>Sex</label>
                  <select
                    id="patient-sex-select"
                    className="input-field select-field"
                    value={patientSex}
                    onChange={(e) => setPatientSex(e.target.value)}
                  >
                    <option value="">Not specified</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div
                style={{
                  background: 'rgba(239,68,68,0.1)',
                  border: '1px solid rgba(239,68,68,0.3)',
                  borderRadius: '10px',
                  padding: '12px 16px',
                  fontSize: '0.85rem',
                  color: '#fca5a5',
                  display: 'flex',
                  gap: '8px',
                  alignItems: 'flex-start',
                }}
              >
                <AlertCircle size={16} style={{ flexShrink: 0, marginTop: '1px' }} />
                {error}
              </div>
            )}

            {/* Analyze Button */}
            <button
              id="analyze-btn"
              className="btn-primary"
              style={{ width: '100%', justifyContent: 'center', fontSize: '1rem', padding: '14px' }}
              onClick={handleAnalyze}
              disabled={!file || loading}
            >
              {loading ? (
                <>
                  <div className="spinner spinner-sm" />
                  Analysing…
                </>
              ) : (
                <>
                  <Scan size={18} />
                  Analyse &amp; Generate Report
                </>
              )}
            </button>

            {/* Reset */}
            {result && (
              <button className="btn-ghost" style={{ justifyContent: 'center' }} onClick={clearFile}>
                <RefreshCw size={14} />
                New Analysis
              </button>
            )}
          </div>

          {/* ── Right Column: Results ──────────────────────── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

            {!result && !loading && (
              <div
                className="glass-card"
                style={{
                  padding: '80px 40px',
                  textAlign: 'center',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '16px',
                  color: 'var(--text-muted)',
                }}
              >
                <div style={{ fontSize: '3rem', opacity: 0.4 }}>🧠</div>
                <div style={{ fontSize: '1rem', fontWeight: 500 }}>
                  Upload an MRI and click Analyse
                </div>
                <div style={{ fontSize: '0.82rem' }}>
                  Results will appear here
                </div>
              </div>
            )}

            {loading && (
              <div
                className="glass-card"
                style={{
                  padding: '80px 40px',
                  textAlign: 'center',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '20px',
                }}
              >
                <div className="spinner" />
                <div style={{ fontWeight: 600, fontSize: '1rem', color: 'var(--text-primary)' }}>
                  Analysing MRI Scan…
                </div>
                <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                  Running preprocessing → classification → severity grading → report generation
                </div>
              </div>
            )}

            {result && (
              <>
                {/* Classification Result */}
                <ResultCard result={result} demoMode={result.demo_mode} />

                {/* Report */}
                <ReportCard
                  report={result.report}
                  tumorClass={result.tumor_class}
                  confidence={result.confidence}
                  severity={result.severity}
                  patientAge={patientAge || null}
                  patientSex={patientSex || null}
                />

                {/* ── Evaluation Panel ────────────────────── */}
                <div className="glass-card" style={{ overflow: 'hidden' }}>
                  <button
                    id="eval-toggle-btn"
                    onClick={() => setEvalOpen(o => !o)}
                    style={{
                      width: '100%',
                      padding: '18px 24px',
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      color: 'var(--text-secondary)',
                    }}
                  >
                    <span style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600, fontSize: '0.9rem' }}>
                      <BarChart2 size={16} color="var(--color-purple)" />
                      NLP Evaluation (BLEU / ROUGE / BERTScore)
                    </span>
                    {evalOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </button>

                  {evalOpen && (
                    <div style={{ padding: '0 24px 24px', borderTop: '1px solid var(--color-border)' }}>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '16px 0 12px' }}>
                        Paste a reference radiology report to compute NLP similarity metrics.
                      </p>
                      <textarea
                        id="reference-report-textarea"
                        className="input-field"
                        rows={5}
                        placeholder="Paste reference radiology report here…"
                        value={refReport}
                        onChange={(e) => setRefReport(e.target.value)}
                        style={{ resize: 'vertical', fontFamily: 'Inter, sans-serif' }}
                      />
                      <button
                        id="evaluate-btn"
                        className="btn-secondary"
                        style={{ marginTop: '12px', width: '100%', justifyContent: 'center' }}
                        onClick={handleEvaluate}
                        disabled={evalLoading || !refReport.trim()}
                      >
                        {evalLoading ? (
                          <><div className="spinner spinner-sm" /> Computing…</>
                        ) : (
                          <><BarChart2 size={15} /> Compute Metrics</>
                        )}
                      </button>

                      {evalResult && !evalResult.error && (
                        <div style={{ marginTop: '16px', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
                          <MetricDisplay label="BLEU-1" value={evalResult.bleu_1} color="var(--color-indigo)" />
                          <MetricDisplay label="ROUGE-L" value={evalResult.rouge_l} color="var(--color-cyan)" />
                          <MetricDisplay label="BERTScore" value={evalResult.bertscore} color="var(--color-purple)" />
                        </div>
                      )}

                      {evalResult?.error && (
                        <div style={{ marginTop: '12px', fontSize: '0.82rem', color: '#fca5a5' }}>
                          {evalResult.error}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricDisplay({ label, value, color }) {
  return (
    <div
      style={{
        background: 'rgba(255,255,255,0.03)',
        borderRadius: '10px',
        padding: '16px',
        textAlign: 'center',
        border: '1px solid var(--color-border)',
      }}
    >
      <div style={{ fontSize: '1.5rem', fontWeight: 700, color, fontFamily: 'Space Grotesk, sans-serif', marginBottom: '4px' }}>
        {typeof value === 'number' ? value.toFixed(4) : (value || '—')}
      </div>
      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
        {label}
      </div>
    </div>
  );
}
