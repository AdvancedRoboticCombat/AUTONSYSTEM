import { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface DashboardPanelProps {
  title: string;
  /** short index shown in the red chip before the title, e.g. "01" */
  tag?: string;
  /** small readout at the right end of the header */
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function DashboardPanel({ title, tag, aside, children, className }: DashboardPanelProps) {
  return (
    <div className={cn('panel-wrap h-full', className)}>
      <div className="panel h-full">
        <div className="panel-body h-full px-3 pt-2.5 pb-3 flex flex-col overflow-hidden">
          <div className="scanline" />
          <div className="relative z-10 flex items-center gap-2 mb-2 flex-shrink-0">
            {tag && <span className="panel-tag">{tag}</span>}
            <h2 className="panel-header text-[13px] whitespace-nowrap">{title}</h2>
            <span className="panel-rule" />
            {aside && <span className="label !text-[11px] whitespace-nowrap pr-3">{aside}</span>}
          </div>
          <div className="flex-1 relative z-10 overflow-auto min-h-0">
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}
