import { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'

const rawApiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const API_BASE = rawApiBase.startsWith('http://') || rawApiBase.startsWith('https://') ? rawApiBase : `https://${rawApiBase}`

const currency = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })
const number = new Intl.NumberFormat('en-US', { maximumFractionDigits: 4 })

const NOTEBOOK_URL = 'https://github.com/panshul-07/sales-forecasting-walmart/tree/main/notebooks'

function KpiCard({ label, value, hint }) {
  return (
    <article className="kpi-card">
      <p className="kpi-label">{label}</p>
      <p className="kpi-value">{value}</p>
      {hint ? <p className="kpi-hint">{hint}</p> : null}
    </article>
  )
}

export default function App() {
  const [stores, setStores] = useState([])
  const [store, setStore] = useState('')
  const [feature, setFeature] = useState('Temperature')

  const [metrics, setMetrics] = useState({})
  const [modelInfo, setModelInfo] = useState(null)

  const [history, setHistory] = useState([])
  const [sensitivity, setSensitivity] = useState([])

  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const [form, setForm] = useState({
    date: new Date().toISOString().slice(0, 10),
    holiday_flag: 0,
    temperature: 70,
    fuel_price: 3.5,
    cpi: 230,
    unemployment: 7,
  })

  async function loadBootstrap() {
    const { data } = await axios.get(`${API_BASE}/api/v1/bootstrap`)
    const storeList = Array.isArray(data?.stores) ? data.stores : []
    setStores(storeList)
    setStore(data?.default_store ? String(data.default_store) : String(storeList[0] || ''))
    setMetrics(data?.metrics || {})
  }

  async function loadModelInfo() {
    const { data } = await axios.get(`${API_BASE}/api/v1/model-info`)
    setModelInfo(data)
  }

  async function loadMetrics() {
    const { data } = await axios.get(`${API_BASE}/api/v1/metrics`)
    setMetrics(data?.metrics || {})
  }

  async function loadHistory(selectedStore) {
    const { data } = await axios.get(`${API_BASE}/api/v1/store/${selectedStore}/history?limit=52`)
    setHistory(Array.isArray(data?.points) ? data.points : [])
  }

  async function loadSensitivity(selectedStore, selectedFeature) {
    const { data } = await axios.get(`${API_BASE}/api/v1/store/${selectedStore}/sensitivity?feature=${selectedFeature}`)
    setSensitivity(Array.isArray(data?.points) ? data.points : [])
  }

  async function initialize() {
    setLoading(true)
    setError('')
    try {
      await loadBootstrap()
      await Promise.all([loadModelInfo(), loadMetrics()])
    } catch (err) {
      setError('Unable to load API data. Check frontend env var VITE_API_BASE_URL and backend service status.')
    } finally {
      setLoading(false)
    }
  }

  async function runTrain() {
    try {
      setError('')
      await axios.post(`${API_BASE}/api/v1/train`)
      await loadMetrics()
      if (store) {
        await Promise.all([loadHistory(store), loadSensitivity(store, feature)])
      }
    } catch (err) {
      setError('Model retraining failed. Try again after backend is fully live.')
    }
  }

  async function runPredict(e) {
    e.preventDefault()
    if (!store) return

    try {
      setError('')
      const payload = {
        ...form,
        store: Number(store),
        holiday_flag: Number(form.holiday_flag),
        temperature: Number(form.temperature),
        fuel_price: Number(form.fuel_price),
        cpi: Number(form.cpi),
        unemployment: Number(form.unemployment),
      }
      const { data } = await axios.post(`${API_BASE}/api/v1/predict`, payload)
      setPrediction(data?.predicted_sales ?? null)
    } catch (err) {
      setError('Prediction failed. Please verify API connectivity and try again.')
    }
  }

  useEffect(() => {
    initialize()
  }, [])

  useEffect(() => {
    if (!store) return
    loadHistory(store).catch(() => setError('Could not load store history.'))
    loadSensitivity(store, feature).catch(() => setError('Could not load sensitivity data.'))
  }, [store, feature])

  const historyAverage = useMemo(() => {
    if (!history.length) return 0
    const sum = history.reduce((acc, p) => acc + Number(p.actual_sales || 0), 0)
    return sum / history.length
  }, [history])

  const changeVsAverage = useMemo(() => {
    if (!prediction || !historyAverage) return null
    return ((prediction - historyAverage) / historyAverage) * 100
  }, [prediction, historyAverage])

  const historyPlot = useMemo(
    () => [
      {
        x: history.map((p) => p.date),
        y: history.map((p) => p.actual_sales),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Actual Sales',
        line: { color: '#0F766E', width: 3 },
        marker: { size: 5, color: '#14B8A6' },
      },
      {
        x: history.map((p) => p.date),
        y: history.map((p) => p.predicted_sales),
        type: 'scatter',
        mode: 'lines',
        name: 'Model Fit',
        line: { color: '#F59E0B', width: 3, dash: 'dash' },
      },
    ],
    [history]
  )

  const sensitivityPlot = useMemo(
    () => [
      {
        x: sensitivity.map((p) => p.x),
        y: sensitivity.map((p) => p.y),
        type: 'scatter',
        mode: 'lines',
        name: `${feature} sensitivity`,
        line: { color: '#7C3AED', width: 3 },
        fill: 'tozeroy',
        fillcolor: 'rgba(124,58,237,0.14)',
      },
    ],
    [sensitivity, feature]
  )

  return (
    <main className="app-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Walmart Forecast Studio</p>
          <h1>Sales Forecasting Dashboard</h1>
          <p className="hero-text">
            Clean forecasting UI for business presentation. Statistical validation and test evidence are documented in notebooks.
          </p>
        </div>
        <div className="hero-actions">
          <button className="btn btn-secondary" onClick={initialize}>Refresh Data</button>
          <button className="btn" onClick={runTrain}>Retrain Model</button>
        </div>
      </section>

      {error ? <div className="alert">{error}</div> : null}

      <section className="control-panel">
        <label>
          Store
          <select value={store} onChange={(e) => setStore(e.target.value)} disabled={loading || stores.length === 0}>
            {stores.length === 0 ? <option value="">No stores loaded</option> : null}
            {stores.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>

        <label>
          Sensitivity Variable
          <select value={feature} onChange={(e) => setFeature(e.target.value)}>
            {['Temperature', 'Fuel_Price', 'CPI', 'Unemployment'].map((f) => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
        </label>

        <a className="btn btn-ghost" href={NOTEBOOK_URL} target="_blank" rel="noreferrer">
          Open Notebook Evidence
        </a>
      </section>

      <section className="kpi-grid">
        <KpiCard label="Ensemble R2" value={number.format(metrics.ensemble_test_r2 || 0)} hint="Holdout accuracy" />
        <KpiCard label="Ensemble RMSE" value={currency.format(metrics.ensemble_test_rmse || 0)} hint="Prediction error" />
        <KpiCard label="Current Forecast" value={currency.format(prediction || 0)} hint="From form inputs" />
        <KpiCard
          label="Change vs Store Avg"
          value={changeVsAverage === null ? '—' : `${changeVsAverage.toFixed(2)}%`}
          hint={historyAverage ? `Store avg ${currency.format(historyAverage)}` : 'Run/load predictions'}
        />
      </section>

      <section className="grid-two">
        <article className="panel">
          <div className="panel-head">
            <h2>Store History</h2>
            <p>Actual vs model fit (last 52 weeks)</p>
          </div>
          <Plot
            data={historyPlot}
            layout={{
              margin: { t: 10, r: 12, b: 40, l: 56 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              font: { color: '#E2E8F0' },
              xaxis: { gridcolor: 'rgba(148,163,184,0.2)' },
              yaxis: { gridcolor: 'rgba(148,163,184,0.2)' },
              legend: { orientation: 'h' },
            }}
            useResizeHandler
            style={{ width: '100%', height: '360px' }}
            config={{ responsive: true, displayModeBar: false }}
          />
        </article>

        <article className="panel">
          <div className="panel-head">
            <h2>{feature} Sensitivity</h2>
            <p>How forecast responds to macro changes</p>
          </div>
          <Plot
            data={sensitivityPlot}
            layout={{
              margin: { t: 10, r: 12, b: 40, l: 56 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              font: { color: '#E2E8F0' },
              xaxis: { gridcolor: 'rgba(148,163,184,0.2)' },
              yaxis: { gridcolor: 'rgba(148,163,184,0.2)' },
            }}
            useResizeHandler
            style={{ width: '100%', height: '360px' }}
            config={{ responsive: true, displayModeBar: false }}
          />
        </article>
      </section>

      <section className="grid-two">
        <article className="panel">
          <div className="panel-head">
            <h2>Forecast Inputs</h2>
            <p>Scenario simulator</p>
          </div>
          <form className="predict-grid" onSubmit={runPredict}>
            <label>
              Date
              <input type="date" value={form.date} onChange={(e) => setForm({ ...form, date: e.target.value })} />
            </label>
            <label>
              Holiday Flag
              <select value={form.holiday_flag} onChange={(e) => setForm({ ...form, holiday_flag: e.target.value })}>
                <option value={0}>0</option>
                <option value={1}>1</option>
              </select>
            </label>
            <label>
              Temperature
              <input type="number" value={form.temperature} onChange={(e) => setForm({ ...form, temperature: e.target.value })} />
            </label>
            <label>
              Fuel Price
              <input type="number" step="0.01" value={form.fuel_price} onChange={(e) => setForm({ ...form, fuel_price: e.target.value })} />
            </label>
            <label>
              CPI
              <input type="number" value={form.cpi} onChange={(e) => setForm({ ...form, cpi: e.target.value })} />
            </label>
            <label>
              Unemployment
              <input type="number" step="0.01" value={form.unemployment} onChange={(e) => setForm({ ...form, unemployment: e.target.value })} />
            </label>
            <button className="btn" type="submit" disabled={!store}>Run Forecast</button>
          </form>
        </article>

        <article className="panel">
          <div className="panel-head">
            <h2>Model Summary</h2>
            <p>Frontend keeps this concise. Full tests remain in notebooks.</p>
          </div>
          <div className="chips">
            <span className="chip">{modelInfo?.forecast_model?.name || 'VotingRegressor'}</span>
            {(modelInfo?.forecast_model?.members || []).map((m) => (
              <span className="chip" key={m}>{m}</span>
            ))}
            <span className="chip">{modelInfo?.interpretable_model?.name || 'OLS'}</span>
            <span className="chip">{modelInfo?.interpretable_model?.target_transform || 'log1p'}</span>
            <span className="chip">{modelInfo?.interpretable_model?.robust_errors || 'HC3'}</span>
          </div>

          <p className="model-note">
            Parametric tests (JB, Shapiro, BG, BP, RESET and others), preprocessing steps, and evidence outputs are documented in the
            notebooks folder for submission.
          </p>

          <a className="btn btn-secondary" href={NOTEBOOK_URL} target="_blank" rel="noreferrer">
            View Notebook + Outputs
          </a>
        </article>
      </section>
    </main>
  )
}
