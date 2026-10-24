import type { ReactNode } from "react";

interface PanelProps {
  title: string;
  /** Right side of the header: counters, badges. */
  meta?: ReactNode;
  className?: string;
  bodyClassName?: string;
  children: ReactNode;
}

export function Panel({ title, meta, className = "", bodyClassName = "", children }: PanelProps) {
  return (
    <section className={`panel ${className}`} aria-label={title}>
      <header className="panel-head">
        <h2 className="label">{title}</h2>
        {meta ? <div className="flex items-center gap-2">{meta}</div> : null}
      </header>
      <div className={`panel-body ${bodyClassName}`}>{children}</div>
    </section>
  );
}
