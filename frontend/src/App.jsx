import { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const currency = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })
const number = new Intl.NumberFormat('en-US', { maximumFractionDigits: 4 })

function Kpi({ label, value }) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
    </div>
  )
}

export default function App() {
  const [stores, setStores] = useState([])
  const [store, setStore] = useState('')
  const [metrics, setMetrics] = useState({})
  const [diagnostics, setDiagnostics] = useState({})
  const [modelInfo, setModelInfo] = useState(null)
  const [history, setHistory] = useState([])
  const [feature, setFeature] = useState('Temperature')
  const [sensitivity, setSensitivity] = useState([])
  const [prediction, setPrediction] = useState(null)

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
    setStores(data.stores)
    setStore(String(data.default_store))
    setMetrics(data.metrics)
  }

  async function loadModelInfo() {
    const { data } = await axios.get(`${API_BASE}/api/v1/model-info`)
    setModelInfo(data)
  }

  async function loadMetrics() {
    const { data } = await axios.get(`${API_BASE}/api/v1/metrics`)
    setMetrics(data.metrics)
    setDiagnostics(data.diagnostics)
  }

  async function loadHistory(selectedStore) {
    const { data } = await axios.get(`${API_BASE}/api/v1/store/${selectedStore}/history?limit=52`)
    setHistory(data.points)
  }

  async function loadSensitivity(selectedStore, selectedFeature) {
    const { data } = await axios.get(`${API_BASE}/api/v1/store/${selectedStore}/sensitivity?feature=${selectedFeature}`)
    setSensitivity(data.points)
  }

  async function runTrain() {
    await axios.post(`${API_BASE}/api/v1/train`)
    await loadMetrics()
    if (store) {
      await loadHistory(store)
      await loadSensitivity(store, feature)
    }
  }

  async function runPredict(e) {
    e.preventDefault()
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
    setPrediction(data.predicted_sales)
  }

  useEffect(() => {
    loadBootstrap().then(loadMetrics)
    loadModelInfo()
  }, [])

  useEffect(() => {
    if (!store) return
    loadHistory(store)
    loadSensitivity(store, feature)
  }, [store, feature])

  const historyTrace = useMemo(
    () => [
      {
        x: history.map((p) => p.date),
        y: history.map((p) => p.actual_sales),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Actual',
        line: { color: '#0A9396', width: 2 },
      },
      {
        x: history.map((p) => p.date),
        y: history.map((p) => p.predicted_sales),
        type: 'scatter',
        mode: 'lines',
        name: 'Predicted',
        line: { color: '#EE9B00', width: 2, dash: 'dash' },
      },
    ],
    [history]
  )

  const sensitivityTrace = useMemo(
    () => [
      {
        x: sensitivity.map((p) => p.x),
        y: sensitivity.map((p) => p.y),
        type: 'scatter',
        mode: 'lines',
        line: { color: '#BB3E03', width: 3 },
        fill: 'tozeroy',
        fillcolor: 'rgba(187, 62, 3, 0.15)',
      },
    ],
    [sensitivity]
  )

  const diagnosticHeatmap = useMemo(() => {
    const labels = ['JB p', 'Shapiro p', 'BP p', 'BG p', 'Ljung-Box p', 'RESET p']
    const values = [
      diagnostics?.normality?.jarque_bera_p ?? 0,
      diagnostics?.normality?.shapiro_p ?? 0,
      diagnostics?.heteroskedasticity?.breusch_pagan_p ?? 0,
      diagnostics?.autocorrelation?.breusch_godfrey_p ?? 0,
      diagnostics?.autocorrelation?.ljung_box_p_lag10 ?? 0,
      diagnostics?.specification?.ramsey_reset_p ?? 0,
    ]
    return [
      {
        z: [values.map((v) => Math.min(6, Math.max(0, -Math.log10(Math.max(v, 1e-12)))))],
        x: labels,
        y: ['-log10(p-value)'],
        type: 'heatmap',
        colorscale: 'YlOrRd',
        zmin: 0,
        zmax: 6,
      },
    ]
  }, [diagnostics])

  return (
    <main>
      <header>
        <h1>Walmart Sales Forecasting</h1>
        <p>FastAPI + React + Plotly dashboard with robust diagnostics, model transparency, and scenario forecasting.</p>
      </header>

      <section className="toolbar">
        <label>
          Store
          <select value={store} onChange={(e) => setStore(e.target.value)}>
            {stores.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <label>
          Sensitivity
          <select value={feature} onChange={(e) => setFeature(e.target.value)}>
            {['Temperature', 'Fuel_Price', 'CPI', 'Unemployment'].map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </label>
        <button onClick={runTrain}>Retrain Model</button>
      </section>

      <section className="kpi-grid">
        <Kpi label="Ensemble Test R2" value={number.format(metrics.ensemble_test_r2 || 0)} />
        <Kpi label="Ensemble RMSE" value={currency.format(metrics.ensemble_test_rmse || 0)} />
        <Kpi label="OLS Test R2" value={number.format(metrics.ols_test_r2 || 0)} />
        <Kpi label="Predicted Sales" value={currency.format(prediction || 0)} />
      </section>

      <section className="panel two-col">
        <div>
          <h2>Model Used</h2>
          <ul className="diag-list">
            <li>Forecast model: {modelInfo?.forecast_model?.name || 'VotingRegressor'}</li>
            <li>Ensemble members: {(modelInfo?.forecast_model?.members || []).join(', ')}</li>
            <li>Interpretability model: {modelInfo?.interpretable_model?.name || 'OLS'}</li>
            <li>Target transform: {modelInfo?.interpretable_model?.target_transform || 'log1p(Weekly_Sales)'}</li>
            <li>Robust errors: {modelInfo?.interpretable_model?.robust_errors || 'HC3'}</li>
          </ul>
        </div>
        <div>
          <h2>Diagnostic Heatmap</h2>
          <Plot
            data={diagnosticHeatmap}
            layout={{ margin: { t: 20, r: 10, b: 70, l: 50 }, paper_bgcolor: '#001219', plot_bgcolor: '#001219', font: { color: '#E9D8A6' } }}
            useResizeHandler
            style={{ width: '100%', height: '260px' }}
            config={{ responsive: true }}
          />
        </div>
      </section>

      <section className="panel">
        <h2>Store History: Actual vs Predicted</h2>
        <Plot
          data={historyTrace}
          layout={{ margin: { t: 20, r: 10, b: 40, l: 60 }, paper_bgcolor: '#001219', plot_bgcolor: '#001219', font: { color: '#E9D8A6' } }}
          useResizeHandler
          style={{ width: '100%', height: '360px' }}
          config={{ responsive: true }}
        />
      </section>

      <section className="panel two-col">
        <div>
          <h2>{feature} Sensitivity</h2>
          <Plot
            data={sensitivityTrace}
            layout={{ margin: { t: 20, r: 10, b: 40, l: 60 }, paper_bgcolor: '#001219', plot_bgcolor: '#001219', font: { color: '#E9D8A6' } }}
            useResizeHandler
            style={{ width: '100%', height: '320px' }}
            config={{ responsive: true }}
          />
        </div>
        <div>
          <h2>OLS Diagnostic Highlights</h2>
          <ul className="diag-list">
            <li>Jarque-Bera p: {number.format(diagnostics?.normality?.jarque_bera_p || 0)}</li>
            <li>Kurtosis: {number.format(diagnostics?.normality?.residual_kurtosis || 0)}</li>
            <li>Breusch-Pagan p: {number.format(diagnostics?.heteroskedasticity?.breusch_pagan_p || 0)}</li>
            <li>Durbin-Watson: {number.format(diagnostics?.autocorrelation?.durbin_watson || 0)}</li>
            <li>RESET p: {number.format(diagnostics?.specification?.ramsey_reset_p || 0)}</li>
            <li>Shapiro p: {number.format(diagnostics?.normality?.shapiro_p || 0)}</li>
          </ul>
        </div>
      </section>

      <section className="panel">
        <h2>Single Prediction</h2>
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
          <button type="submit">Run Prediction</button>
        </form>
      </section>
    </main>
  )
}
