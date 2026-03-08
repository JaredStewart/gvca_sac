import type { CsvColumn } from '@/lib/export-csv'

export interface CompositeScoreRow {
  year: string
  overall: number | null
  grammar: number | null
  middle: number | null
  high: number | null
}

export interface CompositeScoreTableProps {
  title: string
  rows: CompositeScoreRow[]
  weighted: boolean
}

function fmt(val: number | null): string {
  return val != null ? val.toFixed(2) : '-'
}

export default function CompositeScoreTable({ title, rows, weighted }: CompositeScoreTableProps) {
  const sorted = [...rows].sort((a, b) => b.year.localeCompare(a.year))

  return (
    <div>
      <h3 className="text-sm font-semibold mb-1">{title}</h3>
      <p className="text-xs text-muted-foreground mb-2">
        {weighted ? 'Family-weighted (coordinated = 1.0, individual = 0.75)' : 'Unweighted (all responses equal)'}
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b border-border">
              <th className="text-left py-2 px-3 font-semibold">Year</th>
              <th className="text-center py-2 px-3 font-semibold">Overall</th>
              <th className="text-center py-2 px-3 font-semibold">GS</th>
              <th className="text-center py-2 px-3 font-semibold">MS</th>
              <th className="text-center py-2 px-3 font-semibold">HS</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((row) => (
              <tr key={row.year} className="border-b border-border/50 hover:bg-muted/50">
                <td className="py-2 px-3 font-medium">{row.year}</td>
                <td className="text-center py-2 px-3">{fmt(row.overall)}</td>
                <td className="text-center py-2 px-3">{fmt(row.grammar)}</td>
                <td className="text-center py-2 px-3">{fmt(row.middle)}</td>
                <td className="text-center py-2 px-3">{fmt(row.high)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function getCompositeScoreCsvData(rows: CompositeScoreRow[]): Record<string, unknown>[] {
  return [...rows].sort((a, b) => b.year.localeCompare(a.year)).map((r) => ({
    Year: r.year,
    Overall: r.overall,
    GS: r.grammar,
    MS: r.middle,
    HS: r.high,
  }))
}

export const compositeScoreCsvColumns: CsvColumn[] = [
  { header: 'Year', accessor: (r) => r['Year'] as string },
  { header: 'Overall', accessor: (r) => r['Overall'] as number },
  { header: 'GS', accessor: (r) => r['GS'] as number },
  { header: 'MS', accessor: (r) => r['MS'] as number },
  { header: 'HS', accessor: (r) => r['HS'] as number },
]
