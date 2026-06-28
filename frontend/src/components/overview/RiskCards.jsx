export default function RiskCards({ health }) {
  return (
    <section className="risk-stack">
      <article className="panel risk-card fraud">
        <div className="risk-card-head">
          <p>Fraud Alerts</p>
          <span>High Risk</span>
        </div>
        <dl>
          <div>
            <dt>Linked Device ID Matches</dt>
            <dd>42</dd>
          </div>
          <div>
            <dt>VPN / Proxy Detected</dt>
            <dd>188</dd>
          </div>
          <div>
            <dt>Synthetic Identity Flags</dt>
            <dd>12</dd>
          </div>
        </dl>
      </article>
      <article className="panel risk-card sparse">
        <div className="risk-card-head">
          <p>Data Sparsity</p>
          <span>Null Heavy</span>
        </div>
        <dl>
          <div>
            <dt>Missing Experience History</dt>
            <dd>814</dd>
          </div>
          <div>
            <dt>Unverifiable Contact</dt>
            <dd>2,104</dd>
          </div>
          <div>
            <dt>Review Queue Depth</dt>
            <dd>{health?.queue_depth || 18}</dd>
          </div>
        </dl>
      </article>
    </section>
  )
}
