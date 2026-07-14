type Finding = { tag: "positive" | "risk" | "opportunity"; title: string; body: string; action: string };

const TAG_LABEL: Record<Finding["tag"], string> = {
  positive: "Healthy",
  risk: "At risk",
  opportunity: "Opportunity",
};

export function FindingCards({ findings }: { findings: Finding[] }) {
  return (
    <div className="findings">
      {findings.map((f) => (
        <div className={`finding ${f.tag}`} key={f.title}>
          <span className={`ftag ${f.tag}`}>{TAG_LABEL[f.tag]}</span>
          <div className="ftitle">{f.title}</div>
          <p className="fbody">{f.body}</p>
          <div className="faction"><span>→</span> {f.action}</div>
        </div>
      ))}
    </div>
  );
}
