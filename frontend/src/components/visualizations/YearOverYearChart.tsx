import { toStackedData, StackedBarSection, ScaleLegend } from './chart-utils'
import type { CsvColumn } from '@/lib/export-csv'

export interface TrendBar {
  year: string
  distribution: number[]
  counts: number[]
  weightedAverage: number
}

export interface YearOverYearChartProps {
  presentationNumber: number
  shortTitle: string
  scaleLabels: string[]
  trends: TrendBar[]
}

export default function YearOverYearChart({
  presentationNumber,
  shortTitle,
  scaleLabels,
  trends,
}: YearOverYearChartProps) {
  const data = toStackedData(
    trends.map((t) => ({ name: t.year, distribution: t.distribution, weightedAverage: t.weightedAverage }))
  )

  return (
    <div>
      <h3 className="text-sm font-semibold mb-2">
        Q{presentationNumber}: {shortTitle} - Year-Over-Year Trend
      </h3>
      <ScaleLegend scaleLabels={scaleLabels} />
      <StackedBarSection data={data} scaleLabels={scaleLabels} />
    </div>
  )
}

export function getTrendCsvData(
  trends: TrendBar[],
  scaleLabels: string[]
): Record<string, unknown>[] {
  return trends.map((t) => {
    const row: Record<string, unknown> = { Year: t.year }
    for (let i = 0; i < scaleLabels.length; i++) {
      row[`${scaleLabels[i]} (%)`] = t.distribution[i] ?? 0
      row[`${scaleLabels[i]} (N)`] = t.counts[i] ?? 0
    }
    row['Weighted Average'] = t.weightedAverage
    return row
  })
}

export function getTrendCsvColumns(scaleLabels: string[]): CsvColumn[] {
  const cols: CsvColumn[] = [
    { header: 'Year', accessor: (r) => r['Year'] as string },
  ]
  for (const label of scaleLabels) {
    cols.push({ header: `${label} (%)`, accessor: (r) => Number(r[`${label} (%)`]).toFixed(1) })
    cols.push({ header: `${label} (N)`, accessor: (r) => r[`${label} (N)`] as number })
  }
  cols.push({ header: 'Weighted Average', accessor: (r) => Number(r['Weighted Average']).toFixed(2) })
  return cols
}
