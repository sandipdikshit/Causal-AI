# ShieldOS · Real-Time Fraud Decisioning

ShieldOS is a functional front-end prototype for transaction monitoring, explainable risk scoring, automated settlement routing, and fraud-operations traceability.

## Workflow

1. Simulate an incoming transaction using amount, account velocity, device trust, and SIM-swap signals.
2. Inspect each signal's weighted contribution to the aggregate risk score.
3. Route the event to **Approve**, **Step-up Challenge**, or **Hard Block**.
4. Execute the pipeline to append the decision to the audit stream.
5. Select an audit event for investigation or export the current trace as CSV.

The prototype runs entirely in React state. It does not call an external API or persist customer data.

## Run locally

```bash
pnpm install
pnpm dev
```

Open `http://localhost:3000`.

## Validation

```bash
pnpm lint
pnpm build
```

Built with Next.js 16, React 19, TypeScript, Tailwind CSS 4, and Lucide icons.
