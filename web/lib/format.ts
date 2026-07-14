export const usd = (n: number, compact = false) =>
  compact
    ? "$" + Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 }).format(n)
    : "$" + Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(n);

export const num = (n: number) => Intl.NumberFormat("en-US").format(n);

export const pct = (n: number, digits = 1) => `${n.toFixed(digits)}%`;

export const shortMonth = (ym: string) => {
  const [y, m] = ym.split("-");
  const names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${names[parseInt(m, 10) - 1]} ${y.slice(2)}`;
};
